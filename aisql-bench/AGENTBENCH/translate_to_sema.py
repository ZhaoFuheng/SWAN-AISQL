#!/usr/bin/env python3
"""Translate the PLOP agent_bench queries (plop_queries/, PLOP's own dialect) to Sema's (sema_queries/).

PLOP's prompts already name their columns as {t.c} placeholders inside one literal, which is the form of Sema's
s'...' semantic expressions, so the prompt text carries over verbatim:
  SEMANTIC('<text with {t.c}>')       -> s'<text with {t.c}>'   (in WHERE: Sema's semantic filter)
  semantic_int('<text>' || c || '<text>' || d)
                                      -> COALESCE(TRY_CAST(regexp_extract(s'<text>{c}<text>{d}', '^\\s*([-+]?[0-9]+)', 1)
                                         AS INTEGER), 0)   (a semantic map; PLOP's own parse of an integer answer,
                                         as swan_queries/ mirrors it)
  LOAD tpch + CALL dbgen(sf=0.005)    -> views over dataset/tpch_sema.duckdb, attached read-only: the TPC-H tables of
                                         the benchmark's parquet files (dataset/tpch/), which swan_queries/ reads too.
                                         Sema's build has neither the tpch nor the parquet extension;
                                         sema_agentbench.py writes that file with Sema's own shell.
  ./test/semantic/agent_bench/dataset/... -> ./dataset/...
PLOP appends its own answer-format suffix inside its engine; Sema answers in its own format (its system prompt,
and a JSON array when it batches rows), so no suffix is added.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "plop_queries")
DST = os.path.join(HERE, "sema_queries")

TPCH_TABLES = ("part", "supplier", "customer", "lineitem", "orders", "partsupp", "nation", "region")
TPCH_VIEWS = ("-- tpch sf-0.005: the benchmark's parquet files, written into a Sema database by sema_agentbench.py\n"
              "ATTACH IF NOT EXISTS './dataset/tpch_sema.duckdb' AS tpch_sema (READ_ONLY);\n"
              + "\n".join(f"CREATE VIEW IF NOT EXISTS {t} AS SELECT * FROM tpch_sema.{t};" for t in TPCH_TABLES))
INT_PARSE = r"'^\s*([-+]?[0-9]+)'"


def split_literal(sql, start):
    """(literal body with '' kept, index after the closing quote) for the literal at sql[start] == "'"."""
    assert sql[start] == "'"
    i, out = start + 1, []
    while i < len(sql):
        if sql[i] == "'":
            if sql[i + 1:i + 2] == "'":
                out.append("''")
                i += 2
                continue
            return "".join(out), i + 1
        out.append(sql[i])
        i += 1
    raise ValueError("unterminated literal")


def concat_to_template(sql, j, end):
    """The text after the call's first literal, up to its closing paren: ` || col || ' text' || col` -> '{col} text{col}'."""
    rest = sql[j:end].strip()
    out = []
    while rest:
        if not rest.startswith("||"):
            raise ValueError(f"unexpected text in a semantic call: {rest[:40]!r}")
        rest = rest[2:].lstrip()
        if rest.startswith("'"):
            body, k = split_literal(rest, 0)
            out.append(body)
        else:
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?", rest)
            if not m:
                raise ValueError(f"unexpected operand in a semantic call: {rest[:40]!r}")
            out.append("{" + m.group(0) + "}")
            k = m.end()
        rest = rest[k:].lstrip()
    return "".join(out)


def rewrite_calls(sql, fn_pattern, wrap):
    """Replace fn('<literal>' [|| ...]) by wrap(<Sema template>)."""
    out, i = [], 0
    rx = re.compile(fn_pattern, re.IGNORECASE)
    while True:
        m = rx.search(sql, i)
        if not m:
            out.append(sql[i:])
            return "".join(out)
        out.append(sql[i:m.start()])
        j = m.end()
        while sql[j] in " \n\t":
            j += 1
        if sql[j] != "'":
            raise ValueError(f"unexpected argument start: {sql[j:j + 40]!r}")
        body, j = split_literal(sql, j)
        depth, k = 1, j
        while depth:
            if sql[k] == "'":
                _, k = split_literal(sql, k)
                continue
            depth += {"(": 1, ")": -1}.get(sql[k], 0)
            k += 1
        template = body + concat_to_template(sql, j, k - 1)
        if re.search(r"\{(?![A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\})", template):
            raise ValueError("a brace in a prompt that is not a column placeholder")
        out.append(wrap(f"s'{template}'"))
        i = k


def translate(text):
    text = re.sub(r"LOAD\s+'?[^';\n]*tpch[^';\n]*'?;\s*\n?", "", text)
    text = re.sub(r"CALL dbgen\(sf=0\.005\);", TPCH_VIEWS, text)
    text = text.replace("./test/semantic/agent_bench/dataset/", "./dataset/")
    text = rewrite_calls(text, r"\bsemantic_int\s*\(",
                         lambda s: f"COALESCE(TRY_CAST(regexp_extract({s}, {INT_PARSE}, 1) AS INTEGER), 0)")
    text = rewrite_calls(text, r"\bSEMANTIC\s*\(", lambda s: s)
    if re.search(r"\bsemantic(_string|_double)?\s*\(", text, re.IGNORECASE):
        raise ValueError("a semantic function this translator does not handle")
    return text


def main():
    os.makedirs(DST, exist_ok=True)
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


if __name__ == "__main__":
    main()
