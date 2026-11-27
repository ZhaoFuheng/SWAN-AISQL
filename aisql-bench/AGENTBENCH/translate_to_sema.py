#!/usr/bin/env python3
"""Translate the PLOP agent_bench queries (plop_queries/, PLOP's own dialect) to Sema's (sema_queries/).

PLOP's prompts already name their columns as {t.c} placeholders inside one literal, which is the form of Sema's
s'...' semantic expressions, so the prompt text carries over:
  SEMANTIC('<text with {t.c}>')       -> s'<text with {t.c}>'   (in WHERE: Sema's semantic filter)
  semantic_int('<text>' || c || '<text>' || d)
                                      -> COALESCE(TRY_CAST(regexp_extract(s'<text>{c}<text>{d}<integer suffix>',
                                         '^\\s*([-+]?[0-9]+)', 1) AS INTEGER), 0)   (a semantic map, with the
                                         answer-format sentence PLOP's engine appends to a semantic_int prompt and
                                         PLOP's parse of the answer, as swan_queries/ has them)
  LOAD tpch + CALL dbgen(sf=0.005)    -> views over dataset/tpch_sema.duckdb, attached read-only: the TPC-H tables of
                                         the benchmark's parquet files (dataset/tpch/), which swan_queries/ reads too.
                                         Sema's build has neither the tpch nor the parquet extension;
                                         sema_agentbench.py writes that file with Sema's own shell.
  ./test/semantic/agent_bench/dataset/... -> ./dataset/...

Two things in a prompt differ between the engines, and the translation keeps what the model is asked the same:
  - A NULL column. PLOP writes it into the prompt as empty text and asks about every row; Sema asks nothing about a
    row with a NULL placeholder. A SEMANTIC template over a table that has such a NULL (NULL_AS_EMPTY_TEXT) reads
    its columns through a projection that renders them as PLOP does:
      FROM books_info b  ->  FROM books_info b CROSS JOIN LATERAL (SELECT COALESCE(CAST(b.title AS VARCHAR), '')
                             AS title, ...) b_text,   and {b.title} -> {b_text.title}
    The column keeps its name, which is what Sema shows the model. semantic_int is left as it is: PLOP builds that
    prompt with ||, so a NULL column makes the prompt NULL and no row is asked in either engine.
  - An apostrophe. Sema keeps a doubled '' inside s'...' as two characters, so the prompt carries the typographic
    apostrophe instead.
A SEMANTIC answer needs no format sentence: Sema's filter asks for true or false in its own system prompt.
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
#: what PLOP's engine appends to a semantic_int prompt (Morrila src/function/scalar/semantic_functions.cpp)
INT_SUFFIX = "\n Return a single integer, do not contain any other words."
INT_PARSE = r"'^\s*([-+]?[0-9]+)'"
#: the tables with a NULL in a column that a SEMANTIC template reads (dataset/book_review/books_info.csv: subtitle,
#: author, details; dataset/yelp/yelp_business_csv/business.csv: attributes); test_sema.py checks the list
NULL_AS_EMPTY_TEXT = ("books_info", "yelp_business")
APOSTROPHE = "’"
NAME = r"[A-Za-z_][A-Za-z0-9_]*"


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
            m = re.match(rf"{NAME}(?:\.{NAME})?", rest)
            if not m:
                raise ValueError(f"unexpected operand in a semantic call: {rest[:40]!r}")
            out.append("{" + m.group(0) + "}")
            k = m.end()
        rest = rest[k:].lstrip()
    return "".join(out)


def null_as_empty_text(sql):
    """Give every SEMANTIC template over a NULL_AS_EMPTY_TEXT table the projection that renders its columns the way
    PLOP writes them into a prompt (NULL as empty text), and point the template's placeholders at it."""
    edits, projections = [], {}  # (start, end, text) to apply; end of a table reference -> (alias, its columns)
    for m in re.finditer(r"\bSEMANTIC\s*\(\s*", sql, re.IGNORECASE):
        if sql[m.end()] != "'":
            raise ValueError(f"unexpected argument start: {sql[m.end():m.end() + 40]!r}")
        body, end = split_literal(sql, m.end())
        new_body = body
        for alias in sorted({a for a, _ in re.findall(rf"\{{({NAME})\.({NAME})\}}", body)}):
            tables = "|".join(NULL_AS_EMPTY_TEXT)
            declared = list(re.finditer(rf"\bFROM\s+(?:{tables})\s+(?:AS\s+)?{re.escape(alias)}\b", sql[:m.start()],
                                        re.IGNORECASE))
            if not declared:
                continue
            columns = projections.setdefault(declared[-1].end(), (alias, []))[1]
            for a, c in re.findall(rf"\{{({NAME})\.({NAME})\}}", body):
                if a == alias and c not in columns:
                    columns.append(c)
            new_body = re.sub(rf"\{{{re.escape(alias)}\.({NAME})\}}", rf"{{{alias}_text.\1}}", new_body)
        if new_body != body:
            edits.append((m.end(), end, f"'{new_body}'"))
    for position, (alias, columns) in projections.items():
        text = ", ".join(f"COALESCE(CAST({alias}.{c} AS VARCHAR), '') AS {c}" for c in columns)
        edits.append((position, position, f" CROSS JOIN LATERAL (SELECT {text}) {alias}_text"))
    for start, end, text in sorted(edits, reverse=True):
        sql = sql[:start] + text + sql[end:]
    return sql


def rewrite_calls(sql, fn_pattern, wrap, suffix=""):
    """Replace fn('<literal>' [|| ...]) by wrap(<Sema template + suffix>)."""
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
        template = (body + concat_to_template(sql, j, k - 1)).replace("''", APOSTROPHE) + suffix
        if re.search(rf"\{{(?!{NAME}(?:\.{NAME})?\}})", template):
            raise ValueError("a brace in a prompt that is not a column placeholder")
        out.append(wrap(f"s'{template}'"))
        i = k


def translate(text):
    text = re.sub(r"LOAD\s+'?[^';\n]*tpch[^';\n]*'?;\s*\n?", "", text)
    text = re.sub(r"CALL dbgen\(sf=0\.005\);", TPCH_VIEWS, text)
    text = text.replace("./test/semantic/agent_bench/dataset/", "./dataset/")
    text = null_as_empty_text(text)
    text = rewrite_calls(text, r"\bsemantic_int\s*\(",
                         lambda s: f"COALESCE(TRY_CAST(regexp_extract({s}, {INT_PARSE}, 1) AS INTEGER), 0)", INT_SUFFIX)
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
