#!/usr/bin/env python3
"""Translate the Larch govreport {conjunction,disjunction,mix} query JSONs into ai_filter SQL.

Each source query is a fully-parenthesized boolean expression over predicates written as
`(filter_id, "natural-language question")`. We parse that expression and emit SQL that applies each
question to a document's `summary` column via ai_filter, preserving the written (natural) order and
the AND/OR/nesting structure so the reorder-OFF baseline runs the filters as the query has them.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

# How the question is turned into an ai_filter prompt over the document text.
#   ai_filter('<question>' || SEP || summary)  -> yes/no per row
# Single line (no embedded newline) so each emitted query stays on one physical line.
SEP = " Answer yes or no for the following government report: "


class Parser:
    def __init__(self, s):
        self.s = s
        self.i = 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] == " ":
            self.i += 1

    def expect(self, ch):
        self.ws()
        assert self.s[self.i] == ch, f"expected {ch!r} at {self.i}: {self.s[self.i:self.i+20]!r}"
        self.i += 1

    def expr(self):
        # A '(' starts either a leaf `(num, "..")` or a group `( expr OP expr )`.
        self.ws()
        assert self.s[self.i] == "(", f"expected '(' at {self.i}"
        j = self.i + 1
        while j < len(self.s) and self.s[j] == " ":
            j += 1
        if self.s[j].isdigit():
            return self.leaf()
        return self.group()

    def leaf(self):
        self.expect("(")
        self.ws()
        start = self.i
        while self.s[self.i].isdigit():
            self.i += 1
        fid = int(self.s[start:self.i])
        self.expect(",")
        self.expect('"')
        qstart = self.i
        while self.s[self.i] != '"':
            self.i += 1
        question = self.s[qstart:self.i]
        self.i += 1  # closing quote
        self.expect(")")
        return ("leaf", fid, question)

    def group(self):
        self.expect("(")
        left = self.expr()
        self.ws()
        if self.s.startswith("AND", self.i):
            op = "AND"
            self.i += 3
        elif self.s.startswith("OR", self.i):
            op = "OR"
            self.i += 2
        else:
            raise AssertionError(f"expected AND/OR at {self.i}: {self.s[self.i:self.i+10]!r}")
        right = self.expr()
        self.expect(")")
        return (op, left, right)


def leaf_sql(question):
    q = question.replace("'", "''")
    return f"ai_filter('{q}{SEP}' || summary)"


def to_sql(node):
    if node[0] == "leaf":
        return leaf_sql(node[2])
    op, l, r = node
    return f"({to_sql(l)} {op} {to_sql(r)})"


def leaves(node, out):
    if node[0] == "leaf":
        out.append(node[1])
    else:
        leaves(node[1], out)
        leaves(node[2], out)


def main():
    manifest = {}
    for name in ["conjunction", "disjunction", "mix"]:
        d = json.load(open(os.path.join(RAW, f"{name}.json")))
        lines = [
            f"-- govreport {name} queries: {len(d)} boolean ai_filter queries over the `govreport` table.",
            "-- Each query preserves the source expression's order and AND/OR/NOT structure.",
            "",
        ]
        entries = []
        for k in sorted(d, key=lambda x: int(x)):
            ast = Parser(d[k]).expr()
            ls = []
            leaves(ast, ls)
            sql = f"SELECT id FROM govreport WHERE {to_sql(ast)};"
            lines.append(f"-- q{k}: {len(ls)} filters")
            lines.append(sql)
            lines.append("")
            entries.append({"q": int(k), "n_leaves": len(ls), "sql": to_sql(ast)})
        with open(os.path.join(HERE, f"{name}.sql"), "w") as f:
            f.write("\n".join(lines))
        manifest[name] = entries
        print(f"{name}: wrote {len(entries)} queries -> {name}.sql")
    json.dump(manifest, open(os.path.join(HERE, "manifest.json"), "w"), indent=1)
    print("wrote manifest.json")


if __name__ == "__main__":
    main()
