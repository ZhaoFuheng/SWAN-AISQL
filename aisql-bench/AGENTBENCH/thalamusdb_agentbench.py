#!/usr/bin/env python3
"""ThalamusDB on the hybrid bench's 30 queries, in our ThalamusDB formulation (thalamusdb_queries/QN.sql,
written by make_thalamusdb_queries.py; the PLOP authors wrote none), through the same proxy, model and
concurrency as the other systems and SemBench's runner settings for ThalamusDB (exact mode: error bound 0,
call and token caps lifted, 6,000 s per ThalamusDB execution: a staged script runs one per stage and one for
the query). Result frames go to results/thalamusdb/QN.csv (written by
DuckDB, so their formatting matches the other systems' files) and the per-query accounting to
results/thalamusdb_agentbench_results.json; eval_agentbench.py scores them as for every system.

Q1-Q3's classification runs as a CASE cascade of NLfilters (make_thalamusdb_queries.py); ThalamusDB strips a
query's LIMIT and returns every certain row, so the runner keeps the first k (`--apply-limit`), which the
bench's LIMIT scoring requires. Calls and tokens are ThalamusDB's own counters; cost is ESTIMATED at the luna list price (its
requests carry no provider cost header through litellm). ThalamusDB runs in its own interpreter
(../setup_thalamusdb.sh or THALAMUSDB_PYTHON).

  run:  python3 thalamusdb_agentbench.py [Q1 Q2 ...]
  env:  THALAMUSDB_PYTHON, AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna), AI_MAX_CONCURRENCY (20)
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import thalamusdb_common as T  # noqa: E402

TAG = "thalamusdb"


def main():
    which = sys.argv[1:] or [f"Q{i}" for i in range(1, 31)]
    out_dir = os.path.join(HERE, "results", TAG)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(HERE)  # the setup statements read ./dataset/...
    config = T.models_json()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    per_query = {}
    print(f"ThalamusDB agent_bench {T.MODEL} via {T.PROXY}  dop={T.DOP}  max_seconds={T.MAX_SECONDS}  queries={which}\n")
    for name in which:
        script = open(os.path.join(HERE, "thalamusdb_queries", name + ".sql")).read()
        csv_path = os.path.join(out_dir, name + ".csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        _, m0 = T.proxy_stats()
        t0 = time.time()
        res = T.run_query(":memory:", config, script, extra_args=("--csv", csv_path, "--apply-limit"))
        dt = time.time() - t0
        _, m1 = T.proxy_stats()
        err = res["error"]
        if err is None and not os.path.exists(csv_path):
            err = "no result file"  # the exec writes the CSV (header included) whenever the query ran
        cost = T.cost_estimate(res)
        rec = {"latency_s": round(dt, 1), "llm_calls": res["llm_calls"], "llm_calls_fresh": m1 - m0,
               "tokens": res["input_tokens"] + res["output_tokens"], "cost_usd": cost, "cost_is_estimate": True,
               "rows": len(res.get("rows", [])), "error": err, "ran_at": stamp}
        per_query[name] = rec
        print(f"{name:>4} lat={dt:7.1f}s calls={rec['llm_calls']:>6} fresh={rec['llm_calls_fresh']:>6} tok={rec['tokens']:>8} ~${cost:.4f} rows={rec['rows']}"
              + (f"  ERR {err}" if err else ""), flush=True)
        path = os.path.join(HERE, "results", TAG + "_agentbench_results.json")
        old = json.load(open(path)) if os.path.exists(path) else {"per_query": {}}
        old["model"], old["concurrency"], old["updated"] = T.MODEL, T.DOP, stamp
        old["per_query"].update(per_query)
        tot = old["per_query"]
        old["total"] = {"n_queries": len(tot),
                        "latency_s": round(sum(q["latency_s"] for q in tot.values()), 1),
                        "llm_calls": sum(q["llm_calls"] for q in tot.values()),
                        "cost_usd": round(sum(q["cost_usd"] for q in tot.values()), 4), "cost_is_estimate": True,
                        "tokens": sum(q["tokens"] for q in tot.values()),
                        "errors": sum(1 for q in tot.values() if q["error"])}
        json.dump(old, open(path, "w"), indent=1)
    path = os.path.join(HERE, "results", TAG + "_agentbench_results.json")
    if os.path.exists(path):
        print("\nTOTAL", json.load(open(path))["total"])


if __name__ == "__main__":
    main()
