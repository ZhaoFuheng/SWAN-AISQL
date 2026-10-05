#!/usr/bin/env python3
"""ThalamusDB on SemBench MMQA sf_200 -- SemBench's own ThalamusDB queries (thalamusdb_queries/qN.sql, verbatim
from the SemBench repository) on the same data, model, proxy and metrics (precision / recall / F1 against
ground_truth/*.json) as swan_mmqa.py and lotus_mmqa.py, run the way SemBench's runner runs them: ThalamusDB
in exact mode (error bound 0, call and token caps lifted, 6,000 s cap per query), 20 requests in flight.

Of the suite's eleven queries ThalamusDB has a form for seven (q2a, q3a, q3f, q6a, q6b, q6c, q7); q1, q2b,
q4 and q5 need map or summarise operators it does not have and are not run. Tables follow SemBench's
ThalamusDB setup: `movies` = lizzy_caplan_text_data, `tampa_airport` = tampa_international_airport, `images`
with absolute paths. Calls and tokens are ThalamusDB's own counters; cost is ESTIMATED at the luna list price.

  run:  python3 thalamusdb_mmqa.py [q2a q3a ...]
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

TAG = os.environ.get("SWAN_TAG", "")
GT = os.path.join(HERE, "ground_truth")
QUERIES = T.load_queries(os.path.join(HERE, "thalamusdb_queries"))
PAIR_QUERIES = {"q2a", "q7"}  # (id, image file) rows; the rest return one column
CREATE = ("CREATE TABLE ap_warrior AS SELECT * FROM read_parquet('$export/ap_warrior.parquet');\n"
          "CREATE TABLE movies AS SELECT * FROM read_parquet('$export/lizzy_caplan_text_data.parquet');\n"
          "CREATE TABLE tampa_airport AS SELECT * FROM read_parquet('$export/tampa_international_airport.parquet');\n"
          f"CREATE TABLE images AS SELECT row_id, image_filename, '{HERE}/' || regexp_replace(image_filepath, '^\\./', '') AS image_filepath "
          "FROM read_parquet('$export/images.parquet');\n")


def gt(name):
    return json.load(open(os.path.join(GT, name + ".json")))["ground_truth"]


def norm(s):
    return " ".join(str(s).strip().lower().split())


def prf(out, gold):  # as swan_mmqa.py / lotus_mmqa.py
    O = {tuple(norm(v) for v in x) if isinstance(x, tuple) else norm(x) for x in out}
    G = {tuple(norm(v) for v in x) if isinstance(x, (tuple, list)) else norm(x) for x in gold}
    tp = len(O & G)
    P = tp / len(O) if O else (1.0 if not G else 0.0)
    R = tp / len(G) if G else 1.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return round(P, 3), round(R, 3), round(F, 3)


def run_one(name, db, config, stamp):
    _, m0 = T.proxy_stats()
    t0 = time.time()
    res = T.run_query(db, config, QUERIES[name])
    dt = time.time() - t0
    _, m1 = T.proxy_stats()
    rows = res["rows"]
    out = [(T.cell(r[0]), T.cell(r[1])) for r in rows] if name in PAIR_QUERIES else [T.cell(r[0]) for r in rows]
    gold = gt(name)
    err = res["error"]
    P, R, F = prf(out, gold) if err is None else (0.0, 0.0, 0.0)
    cost = T.cost_estimate(res)
    rec = {"latency_s": round(dt, 1), "llm_calls_fresh": m1 - m0, "llm_calls": res["llm_calls"], "cache_hits": None,
           "tokens": res["input_tokens"] + res["output_tokens"], "reasoning_tokens": None,
           "cost_usd": cost, "cost_is_estimate": True,
           "n_out": len(out), "n_gold": len(gold), "precision": P, "recall": R, "f1": F, "quality": F,
           "error": err, "ran_at": stamp}
    raw_entry = {"predicted": [list(x) if isinstance(x, tuple) else x for x in out],
                 "gold": [list(x) if isinstance(x, (tuple, list)) else x for x in gold]}
    tag = f"ERR {err}" if err else f"P={P} R={R} F1={F}"
    line = (f"{name:5} lat={rec['latency_s']:6.1f}s calls={rec['llm_calls']:>5} fresh={rec['llm_calls_fresh']:4} "
            f"tok={rec['tokens']:>7} ~${cost:.4f}  n={rec['n_out']}/{rec['n_gold']}  {tag}")
    return rec, raw_entry, line


def main():
    which = sys.argv[1:] or list(QUERIES)
    RES, RAW, LOG = (os.path.join(HERE, f) for f in
                     (f"thalamusdb_mmqa_results{TAG}.json", f"thalamusdb_mmqa_raw{TAG}.json", f"thalamusdb_mmqa_log{TAG}.txt"))

    def _load(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    per_query = _load(RES).get("per_query", {})
    raw = _load(RAW)
    db = T.build_db(HERE, os.path.join(HERE, "mmqa.db"),
                    ["ap_warrior", "lizzy_caplan_text_data", "tampa_international_airport", "images"], CREATE)
    config = T.models_json()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header = f"ThalamusDB MMQA {T.MODEL} via {T.PROXY}  dop={T.DOP}  max_seconds={T.MAX_SECONDS}  queries={which}"
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
           "macro_f1": round(sum(r["f1"] for r in allq.values()) / len(allq), 3)}
    tline = (f"TOTAL({len(allq)}q)  latency={tot['latency_s']}s  calls={tot['llm_calls']}  ~cost=${tot['cost_usd']}  "
             f"tokens={tot['tokens']}  macro-F1={tot['macro_f1']}")
    print("\n" + tline)
    json.dump({"model": T.MODEL, "concurrency": T.DOP, "updated": stamp, "per_query": per_query, "total": tot},
              open(RES, "w"), indent=1)
    json.dump(raw, open(RAW, "w"), indent=1)
    with open(LOG, "a") as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + tline + "\n\n")


if __name__ == "__main__":
    main()
