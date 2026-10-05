#!/usr/bin/env python3
"""Palimpzest (with its Abacus optimizer on) on SemBench MMQA sf_200 -- SemBench's own Palimpzest programs
(palimpzest_queries.py, copied from the SemBench repository; all eleven queries) on the same data, model,
proxy and metrics (precision / recall / F1 against ground_truth/*.json) as swan_mmqa.py and lotus_mmqa.py,
under SemBench's Palimpzest settings with the optimizer on (../palimpzest_common.py).

Calls, fresh calls and tokens are the proxy's counter deltas while the query runs (run with nothing else
using the proxy); cost is ESTIMATED from those tokens at the luna list price.

  run:  python3 palimpzest_mmqa.py [q1 q2a ...]
  env:  PALIMPZEST_PYTHON, AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna), AI_MAX_CONCURRENCY (20),
        PALIMPZEST_OPTIMIZER (pareto | none), PALIMPZEST_POLICY (MaxQuality | MinCost)
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import palimpzest_common as P  # noqa: E402

TAG = os.environ.get("SWAN_TAG", "")
GT = os.path.join(HERE, "ground_truth")
QUERIES = ["q1", "q2a", "q2b", "q3a", "q3f", "q4", "q5", "q6a", "q6b", "q6c", "q7"]
MODULE = os.path.join(HERE, "palimpzest_queries.py")
DATA = os.path.join(HERE, "files", "mmqa", "data", "sf_200")


def gt(name):
    g = json.load(open(os.path.join(GT, name + ".json")))["ground_truth"]
    if name == "q4":
        return [(genre, movie) for genre, movies in g.items() for movie in movies]
    return [tuple(x) if isinstance(x, list) else x for x in g]


def norm(s):
    return " ".join(str(s).strip().lower().split())


def prf(out, gold):  # as swan_mmqa.py / lotus_mmqa.py
    O = {tuple(norm(v) for v in x) if isinstance(x, tuple) else norm(x) for x in out}
    G = {tuple(norm(v) for v in x) if isinstance(x, tuple) else norm(x) for x in gold}
    tp = len(O & G)
    Pq = tp / len(O) if O else (1.0 if not G else 0.0)
    R = tp / len(G) if G else 1.0
    F = 2 * Pq * R / (Pq + R) if (Pq + R) else 0.0
    return round(Pq, 3), round(R, 3), round(F, 3)


def extract(name, res):
    cols, rows = res["columns"], res["rows"]
    col = lambda c: cols.index(c)  # noqa: E731
    if name in ("q2a", "q7"):
        return [(P.cell(r[col(cols[0])]), P.cell(r[col("filename")])) for r in rows]
    if name == "q2b":
        return [(P.cell(r[0]), P.cell(r[col("filename")]), P.cell(r[col("color")])) for r in rows]
    if name == "q4":  # SemBench's table (genre, "title, title, ...") -> (genre, title) pairs
        return [(P.cell(r[col("genre")]), t.strip()) for r in rows for t in str(r[col("movies_in_genre")]).split(", ") if t.strip()]
    return [P.cell(r[0]) for r in rows]


def run_one(name, stamp):
    t0 = time.time()  # wall time of the whole subprocess, interpreter start-up included; latency_s is the query's own time
    res = P.run_query(MODULE, name, DATA)
    dt = time.time() - t0
    err = res["error"]
    try:
        out = extract(name, res) if err is None else []
    except Exception as ex:  # noqa: BLE001
        out, err = [], f"output: {ex!r}"[:200]
    gold = gt(name)
    Pq, R, F = prf(out, gold) if err is None else (0.0, 0.0, 0.0)
    # latency_s = Palimpzest's own execution time of the plan (its execution_stats), wall_s = the subprocess
    rec = {"latency_s": round(res.get("stats", {}).get("total_execution_time", res["seconds"]), 1), "wall_s": round(dt, 1), "llm_calls_fresh": res["llm_calls_fresh"], "llm_calls": res["llm_calls"], "cache_hits": None,
           "tokens": res["tokens"], "reasoning_tokens": None, "cost_usd": res["cost_usd"], "cost_is_estimate": True,
           "optimizer": P.OPTIMIZER, "policy": P.POLICY, "pz_stats": res.get("stats", {}),
           "n_out": len(out), "n_gold": len(gold), "precision": Pq, "recall": R, "f1": F, "quality": F,
           "error": err, "ran_at": stamp}
    raw_entry = {"predicted": [list(x) if isinstance(x, tuple) else x for x in out],
                 "gold": [list(x) if isinstance(x, tuple) else x for x in gold]}
    tag = f"ERR {err}" if err else f"P={Pq} R={R} F1={F}"
    line = (f"{name:5} lat={rec['latency_s']:6.1f}s calls={rec['llm_calls']:>5} fresh={rec['llm_calls_fresh']:4} "
            f"tok={rec['tokens']:>7} ~${rec['cost_usd']:.4f}  n={rec['n_out']}/{rec['n_gold']}  {tag}")
    return rec, raw_entry, line


def main():
    which = sys.argv[1:] or QUERIES
    RES, RAW, LOG = (os.path.join(HERE, f) for f in
                     (f"palimpzest_mmqa_results{TAG}.json", f"palimpzest_mmqa_raw{TAG}.json", f"palimpzest_mmqa_log{TAG}.txt"))

    def _load(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    per_query = _load(RES).get("per_query", {})
    raw = _load(RAW)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header = f"Palimpzest MMQA {P.MODEL} via {P.PROXY}  optimizer={P.OPTIMIZER}  policy={P.POLICY}  workers={P.DOP}  queries={which}"
    print(header + "\n")
    lines = []
    for name in which:
        rec, raw_entry, line = run_one(name, stamp)
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
    json.dump({"model": P.MODEL, "optimizer": P.OPTIMIZER, "policy": P.POLICY, "concurrency": P.DOP, "updated": stamp,
               "per_query": per_query, "total": tot}, open(RES, "w"), indent=1)
    json.dump(raw, open(RAW, "w"), indent=1)
    with open(LOG, "a") as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + tline + "\n\n")


if __name__ == "__main__":
    main()
