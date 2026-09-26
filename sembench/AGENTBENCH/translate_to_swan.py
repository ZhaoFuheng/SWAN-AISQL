#!/usr/bin/env python3
"""Translate the PLOP agent_bench queries (Morrila-8DBC) to SWAN's operator surface.

PROMPT-IDENTICAL mode: every semantic op goes through ai_complete with PLOP's exact prompt
(their appended format suffix included) and PLOP's exact client-side parse mirrored in SQL, so
the two systems' calls differ only in transport (chat vs Responses API) and ai_complete's
neutral 2-line system prompt. This isolates placement/execution differences from prompt
engineering in the quality scores:
  SEMANTIC(p)        -> ai_filter(p)        (boolean predicate <-> boolean predicate)
  semantic_int(p)    -> COALESCE(TRY_CAST(regexp_extract(ai_complete(p || <int suffix>),
                        '^\s*([-+]?[0-9]+)', 1) AS INTEGER), 0)                       [stoll]
  semantic_string(p) -> ai_complete(p || <string suffix>)
  semantic_double(p) -> COALESCE(TRY_CAST(regexp_extract(...double suffix...) AS DOUBLE), 0.0)
  LOAD tpch + dbgen(sf=0.005)   -> views over dataset/tpch/*.parquet (dbgen-exported, canonical)
  ./test/semantic/agent_bench/dataset/... -> ./dataset/... (runner cwd = AGENTBENCH)
"""

BOOL_SUF = "\n Return a single yes or no, do not contain any other words."
INT_SUF = "\n Return a single integer, do not contain any other words."
STR_SUF = "\n Return a single word or phrase, do not contain any other words."
DBL_SUF = "\n Return a single floating point number, do not contain any other words."


def sql_lit(s):
    return "e'" + s.replace("\\", "\\\\").replace("'", "''").replace("\n", "\\n") + "'"
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "plop_queries")
DST = os.path.join(HERE, "swan_queries")
os.makedirs(DST, exist_ok=True)

TPCH_VIEWS = """-- tpch sf-0.005 from the benchmark's committed parquet files
CREATE VIEW IF NOT EXISTS part     AS SELECT * FROM './dataset/tpch/part.parquet';
CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM './dataset/tpch/supplier.parquet';
CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM './dataset/tpch/customer.parquet';
CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM './dataset/tpch/lineitem.parquet';
CREATE VIEW IF NOT EXISTS orders   AS SELECT * FROM './dataset/tpch/orders.parquet';
CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM './dataset/tpch/partsupp.parquet';
CREATE VIEW IF NOT EXISTS nation   AS SELECT * FROM './dataset/tpch/nation.parquet';
CREATE VIEW IF NOT EXISTS region   AS SELECT * FROM './dataset/tpch/region.parquet';"""


def split_literal(sql, start):
    """Return (literal_body, end_index_after_quote) for the single-quoted literal at sql[start]=="'".
    Handles '' escapes."""
    assert sql[start] == "'"
    i = start + 1
    out = []
    while i < len(sql):
        if sql[i] == "'":
            if i + 1 < len(sql) and sql[i + 1] == "'":
                out.append("''")
                i += 2
                continue
            return "".join(out), i + 1
        out.append(sql[i])
        i += 1
    raise ValueError("unterminated literal")


def placeholders_to_concat(body):
    """'a {t.c} b' -> 'a ' || t.c || ' b' (literal text kept verbatim, incl. newlines)."""
    parts = re.split(r"\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\}", body)
    if len(parts) == 1:
        return "'" + body + "'"
    pieces = []
    for i, p in enumerate(parts):
        if i % 2 == 0:
            if p:
                pieces.append("'" + p + "'")
        else:
            pieces.append(p)
    return " || ".join(pieces)


def rewrite_calls(sql, fn_pattern, wrap):
    """Replace fn('literal', ...) with wrap(translated_literal). Only single-literal calls occur."""
    out = []
    i = 0
    rx = re.compile(fn_pattern, re.IGNORECASE)
    while True:
        m = rx.search(sql, i)
        if not m:
            out.append(sql[i:])
            return "".join(out)
        out.append(sql[i:m.start()])
        j = m.end()  # positioned right after the opening '('
        while sql[j] in " \n\t":
            j += 1
        if sql[j] != "'":
            raise ValueError(f"unexpected arg start at {j}: {sql[j:j+40]!r}")
        body, j = split_literal(sql, j)
        # the remainder up to the matching ')' may contain ' || col || ...' concat args
        depth = 1
        k = j
        while depth:
            if sql[k] == "'":
                _, k = split_literal(sql, k)
                continue
            if sql[k] == "(":
                depth += 1
            elif sql[k] == ")":
                depth -= 1
            k += 1
        tail = sql[j:k - 1]  # verbatim (concat chains already SQL)
        expr = placeholders_to_concat(body) + tail
        out.append(wrap(expr))
        i = k


def translate(text):
    # dbgen block -> parquet views
    text = re.sub(r"LOAD\s+'?[^';\n]*tpch[^';\n]*'?;\s*\n?", "", text)
    text = re.sub(r"CALL dbgen\(sf=0\.005\);", TPCH_VIEWS, text)
    text = text.replace("./test/semantic/agent_bench/dataset/", "./dataset/")
    # PLOP's SEMANTIC() IS a boolean predicate, so ai_filter is the faithful counterpart (same
    # principle as semantic_int -> ai_classify). It is also the form SWAN's optimizer recognises:
    # AIDetectMixedLeaf classifies ai_filter as an AI leaf, so the semantic pull-up, the DP
    # reorder and the embedding-based selectivity model all apply. The nested
    # lower(ai_complete(..)) IN (..) spelling is NOT recognised, which left the predicate pinned
    # to its base table (agent_bench Q22: 89 calls where 10 rows survive; Q26: 1,445 where 5 do).
    # PLOP appends BOOL_SUF to the prompt INSIDE semantic(), so its recorded sample is keyed on
    # prompt+suffix. Dropping the suffix here made every ai_filter call miss the shared-verdict
    # alias and resample the verdict independently -- which is what the macro-F1 gap measured
    # (0.801 vs 0.972 on the ai_complete spelling), not a placement or execution difference.
    text = rewrite_calls(text, r"\bSEMANTIC\s*\(", lambda e: f"ai_filter({e} || {sql_lit(BOOL_SUF)})")
    text = rewrite_calls(text, r"\bsemantic_string\s*\(",
                         lambda e: f"ai_complete({e} || {sql_lit(STR_SUF)})")
    text = rewrite_calls(
        text, r"\bsemantic_int\s*\(",
        lambda e: f"COALESCE(TRY_CAST(regexp_extract(ai_complete({e} || {sql_lit(INT_SUF)}), "
                  f"'^\\s*([-+]?[0-9]+)', 1) AS INTEGER), 0)")
    text = rewrite_calls(
        text, r"\bsemantic_double\s*\(",
        lambda e: f"COALESCE(TRY_CAST(regexp_extract(ai_complete({e} || {sql_lit(DBL_SUF)}), "
                  f"'^\\s*([-+]?[0-9]*\\.?[0-9]+)', 1) AS DOUBLE), 0.0)")
    return text


def main():
    fails = []
    for f in sorted(os.listdir(SRC)):
        if not f.endswith(".sql"):
            continue
        try:
            t = translate(open(os.path.join(SRC, f)).read())
            open(os.path.join(DST, f), "w").write(t)
        except Exception as e:
            fails.append((f, repr(e)))
    print(f"translated {len(os.listdir(DST))} files; {len(fails)} failures")
    for f, e in fails:
        print(" FAIL", f, e)
    sys.exit(1 if fails else 0)


main()
