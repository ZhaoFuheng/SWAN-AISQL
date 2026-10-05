#!/usr/bin/env python3
"""ThalamusDB on SemBench ECOMM sf_500 -- SemBench's own ThalamusDB queries (thalamusdb_queries/qN.sql, from the
SemBench repository) on the same data, model, proxy and metrics (ecomm_common) as swan_ecomm.py and
lotus_ecomm.py, run the way SemBench's runner runs them: ThalamusDB in exact mode (error bound 0, call and
token caps lifted, 6,000 s cap per query), 20 requests in flight, its own join batching. ThalamusDB takes
images, so it runs the text-and-image queries SemBench wrote for it.

SemBench runs five of the fourteen queries on ThalamusDB (q1, q2, q7, q8, q9); the others need map,
classify or ranking operators it does not have. The tables follow SemBench's ThalamusDB setup:
`styles_details` with `full_product_description` (title and description concatenated, single quotes
removed) and `image_mapping` with `local_image_path`. Calls and tokens are ThalamusDB's own counters; cost
is ESTIMATED at the luna list price.

  run:  python3 thalamusdb_ecomm.py [q1 q2 ...]
  env:  THALAMUSDB_PYTHON (see ../setup_thalamusdb.sh), AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna),
        AI_MAX_CONCURRENCY (20), DUCKDB_BIN
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import thalamusdb_common as T  # noqa: E402
from ecomm_common import gt, score  # noqa: E402

TAG = os.environ.get("SWAN_TAG", "")
QUERIES = T.load_queries(os.path.join(HERE, "thalamusdb_queries"))
IMAGES = os.path.join(HERE, "data", "sf_500", "images", "")
CREATE = ("CREATE TABLE styles_details AS SELECT *, replace(title || ' ' || description, '''', '') AS full_product_description "
          "FROM read_parquet('$export/styles_details.parquet');\n"
          f"CREATE TABLE image_mapping AS SELECT id, filename, '{IMAGES}' || filename AS local_image_path "
          "FROM read_parquet('$export/images.parquet');\n")


def run_one(name, db, config, stamp):
    _, m0 = T.proxy_stats()
    t0 = time.time()
    res = T.run_query(db, config, QUERIES[name])
    dt = time.time() - t0
    _, m1 = T.proxy_stats()
    out = [T.cell(row[0]) for row in res["rows"]]  # every ThalamusDB query here returns one id column
    gold = gt(name)
    err = res["error"]
    P, R, M = score(name, out, gold) if err is None else (0.0, 0.0, 0.0)
    cost = T.cost_estimate(res)
    rec = {"latency_s": round(dt, 1), "llm_calls_fresh": m1 - m0, "llm_calls": res["llm_calls"], "cache_hits": None,
           "tokens": res["input_tokens"] + res["output_tokens"], "reasoning_tokens": None,
           "cost_usd": cost, "cost_is_estimate": True,
           "n_out": len(out), "n_gold": len(gold), "precision": P, "recall": R, "metric": "f1",
           "quality": M, "error": err, "ran_at": stamp}
    raw_entry = {"predicted": out[:3000]}
    tag = f"ERR {err}" if err else f"P={P} R={R} quality={M}"
    line = (f"{name:4} lat={rec['latency_s']:7.1f}s calls={rec['llm_calls']:>6} fresh={rec['llm_calls_fresh']:>5} "
            f"tok={rec['tokens']:>9} ~${cost:.4f}  n={rec['n_out']}/{rec['n_gold']}  {tag}")
    return rec, raw_entry, line


def main():
    which = sys.argv[1:] or list(QUERIES)
    RES, RAW, LOG = (os.path.join(HERE, f) for f in
                     (f"thalamusdb_ecomm_results{TAG}.json", f"thalamusdb_ecomm_raw{TAG}.json", f"thalamusdb_ecomm_log{TAG}.txt"))

    def _load(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    per_query = _load(RES).get("per_query", {})
    raw = _load(RAW)
    db = T.build_db(HERE, os.path.join(HERE, "ecomm.db"), ["styles_details", "images"], CREATE)
    config = T.models_json()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header = f"ThalamusDB ECOMM {T.MODEL} via {T.PROXY}  dop={T.DOP}  max_seconds={T.MAX_SECONDS}  sf500  serial  queries={which}"
    print(header + "\n")
    lines = []
    for name in which:
        rec, raw_entry, line = run_one(name, db, config, stamp)
        per_query[name], raw[name] = rec, raw_entry
        print(line, flush=True)
        lines.append(line)
    allq = per_query
    tot = {"n_queries": len(allq),
           "latency_s": round(sum(r["latency_s"] for r in allq.values()), 1),
           "cost_usd": round(sum(r["cost_usd"] for r in allq.values()), 4), "cost_is_estimate": True,
           "tokens": sum(r.get("tokens") or 0 for r in allq.values()),
           "llm_calls": sum(r.get("llm_calls") or 0 for r in allq.values()),
           "macro_quality": round(sum(r["quality"] for r in allq.values()) / len(allq), 3)}
    tline = (f"TOTAL({len(allq)}q)  latency={tot['latency_s']}s  calls={tot['llm_calls']}  ~cost=${tot['cost_usd']}  "
             f"tokens={tot['tokens']}  macro-quality={tot['macro_quality']}")
    print("\n" + tline)
    json.dump({"model": T.MODEL, "concurrency": T.DOP, "updated": stamp, "per_query": per_query, "total": tot},
              open(RES, "w"), indent=1)
    json.dump(raw, open(RAW, "w"), indent=1)
    with open(LOG, "a") as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + tline + "\n\n")


if __name__ == "__main__":
    main()
