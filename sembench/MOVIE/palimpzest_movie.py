#!/usr/bin/env python3
"""Palimpzest (with its Abacus optimizer on) on SemBench MOVIE sf_2000 -- SemBench's own Palimpzest programs
(palimpzest_queries.py, copied from the SemBench repository) on the same data, model, proxy and metrics
(movie_common) as swan_movie.py and lotus_movie.py, under SemBench's Palimpzest settings (MaxQuality, parallel
execution, 20 workers) with the optimizer on, which SemBench had off (../palimpzest_common.py).

Calls, fresh calls and tokens are the proxy's counter deltas while the query runs, so run this with nothing
else using the proxy; cost is ESTIMATED from those tokens at the luna list price. Abacus's sampling calls
count as the query's calls.

  run:  python3 palimpzest_movie.py [q1 q2 ...]
  env:  PALIMPZEST_PYTHON (see ../setup_palimpzest.sh), AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna),
        AI_MAX_CONCURRENCY (20), PALIMPZEST_OPTIMIZER (pareto = Abacus on | none), PALIMPZEST_POLICY (MaxQuality | MinCost)
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import palimpzest_common as P  # noqa: E402
from movie_common import AGG_QUERIES, gt, metric_fields, metric_tag, score  # noqa: E402

TAG = os.environ.get("SWAN_TAG", "")
QUERIES = [f"q{i}" for i in range(1, 11)]
MODULE = os.path.join(HERE, "palimpzest_queries.py")
DATA = os.path.join(HERE, "data", "sf_2000")


def run_one(name, stamp):
    t0 = time.time()  # wall time of the whole subprocess, interpreter start-up included; latency_s is the query's own time
    res = P.run_query(MODULE, name, DATA)
    dt = time.time() - t0
    out = [tuple(P.cell(v, integral_as_int=name not in AGG_QUERIES) for v in row) for row in res["rows"]]
    gold = gt(name)
    err = res["error"]
    Pq, R, M = score(name, out, gold) if err is None else (0.0, 0.0, 0.0)
    # latency_s = Palimpzest's own execution time of the plan (its execution_stats), wall_s = the subprocess
    rec = {"latency_s": round(res.get("stats", {}).get("total_execution_time", res["seconds"]), 1), "wall_s": round(dt, 1), "llm_calls_fresh": res["llm_calls_fresh"], "llm_calls": res["llm_calls"], "cache_hits": None,
           "tokens": res["tokens"], "reasoning_tokens": None, "cost_usd": res["cost_usd"], "cost_is_estimate": True,
           "optimizer": P.OPTIMIZER, "policy": P.POLICY, "pz_stats": res.get("stats", {}),
           "n_out": len(out), "n_gold": len(gold), **metric_fields(name, Pq, R),
           "quality": M, "error": err, "ran_at": stamp}
    raw_entry = {"predicted": [list(x) for x in out][:5000]}
    tag = f"ERR {err}" if err else f"{metric_tag(name, Pq, R)} quality={M}"
    line = (f"{name:4} lat={rec['latency_s']:7.1f}s calls={rec['llm_calls']:>6} fresh={rec['llm_calls_fresh']:>5} "
            f"tok={rec['tokens']:>9} ~${rec['cost_usd']:.4f}  n={rec['n_out']}/{rec['n_gold']}  {tag}")
    return rec, raw_entry, line


def main():
    which = sys.argv[1:] or QUERIES
    RES, RAW, LOG = (os.path.join(HERE, f) for f in
                     (f"palimpzest_movie_results{TAG}.json", f"palimpzest_movie_raw{TAG}.json", f"palimpzest_movie_log{TAG}.txt"))

    def _load(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    per_query = _load(RES).get("per_query", {})
    raw = _load(RAW)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header = f"Palimpzest MOVIE {P.MODEL} via {P.PROXY}  optimizer={P.OPTIMIZER}  policy={P.POLICY}  workers={P.DOP}  sf2000  serial  queries={which}"
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
           "macro_quality": round(sum(r["quality"] for r in allq.values()) / len(allq), 3)}
    tline = (f"TOTAL({len(allq)}q)  latency={tot['latency_s']}s  calls={tot['llm_calls']}  ~cost=${tot['cost_usd']}  "
             f"tokens={tot['tokens']}  macro-quality={tot['macro_quality']}")
    print("\n" + tline)
    json.dump({"model": P.MODEL, "optimizer": P.OPTIMIZER, "policy": P.POLICY, "concurrency": P.DOP, "updated": stamp,
               "per_query": per_query, "total": tot}, open(RES, "w"), indent=1)
    json.dump(raw, open(RAW, "w"), indent=1)
    with open(LOG, "a") as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + tline + "\n\n")


if __name__ == "__main__":
    main()
