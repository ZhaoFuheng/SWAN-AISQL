#!/usr/bin/env python3
"""Generate gold/qN.csv from the vendored SemBench gold SQL (exact relational queries over the CSVs).

Uses the fork's duckdb binary; the gold SQL references tables Movies and Reviews, provided as views
over data/sf_2000/*.csv.  Run once:  python3 gen_ground_truth.py
"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.abspath(os.path.join(HERE, "../../build/release/duckdb"))
DATA = os.path.join(HERE, "data", "sf_2000")

VIEWS = (f"CREATE VIEW Movies AS SELECT * FROM read_csv('{DATA}/Movies.csv');"
         f"CREATE VIEW Reviews AS SELECT * FROM read_csv('{DATA}/Reviews.csv');")

os.makedirs(os.path.join(HERE, "gold"), exist_ok=True)
for i in range(1, 11):
    sql = open(os.path.join(HERE, "queries", "gold_sql", f"Q{i}.sql")).read().strip().rstrip(";")
    out = os.path.join(HERE, "gold", f"q{i}.csv")
    script = f"{VIEWS}\n.mode csv\n.headers on\n.once {out}\n{sql};\n"
    p = subprocess.run([BIN], input=script, text=True, capture_output=True)
    n = sum(1 for _ in open(out)) - 1 if os.path.exists(out) else -1
    print(f"q{i}: {n} gold rows" + (f"  ERR {p.stderr.strip()[:120]}" if p.returncode else ""))
