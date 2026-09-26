#!/usr/bin/env python3
"""Generate ground_truth/qN.json by running each TOML's exact relational ground-truth SQL on sf_500.

The SemBench TOMLs ship a plain-SQL ground truth over the raw parquets (read_parquet with RELATIVE paths),
so this runs DuckDB with cwd = the data dir. ARI queries (q3-q6) emit (id, category) pairs; the rest an id list.

  usage: python3 gen_ground_truth.py [data_dir]   (default: ./data/sf_500)
"""
import json
import os
import sys
import tomllib

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "sf_500"))
OUT = os.path.join(HERE, "ground_truth")
ARI = {"q3", "q4", "q5", "q6"}

os.makedirs(OUT, exist_ok=True)
os.chdir(DATA)  # ground-truth SQL uses relative read_parquet paths
for n in range(1, 15):
    name = f"q{n}"
    spec = tomllib.load(open(os.path.join(HERE, "toml", name + ".toml"), "rb"))
    sql = spec["definition"]["ground_truth"]
    rows = duckdb.sql(sql).fetchall()
    if name in ARI:
        val = [[str(r[0]), str(r[1])] for r in rows]
    else:
        val = [str(r[0]) for r in rows]
    json.dump({"ground_truth": val, "metric": spec["definition"]["accuracy_metric"]},
              open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(f"{name}: {len(val)} gold rows ({spec['definition']['accuracy_metric']})")
