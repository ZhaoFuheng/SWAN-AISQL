#!/usr/bin/env python3
"""Palimpzest (under its Abacus optimizer) on the hybrid bench's 30 queries: the bench's LOTUS programs
(lotus_queries/QN.py, our translation of the SWAN queries) run unchanged with their semantic operators served
by Palimpzest (../palimpzest_hybrid_exec.py gives pandas frames `sem_filter` / `sem_map` backed by Palimpzest
programs), through the same proxy, model and concurrency as the other systems. Result frames go to
results/palimpzest/QN.csv (written with pandas, as LOTUS's are) and the per-query accounting to
results/palimpzest_agentbench_results.json; eval_agentbench.py scores them as for every system.

Calls, fresh calls and tokens are the proxy's counter deltas while a query runs, so run this with nothing
else using the proxy; cost is ESTIMATED from those tokens at the luna list price. Palimpzest runs in its own interpreter (../setup_palimpzest.sh or PALIMPZEST_PYTHON).

  run:  python3 palimpzest_agentbench.py [Q1 Q2 ...]
  env:  PALIMPZEST_PYTHON, AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna), AI_MAX_CONCURRENCY (20),
        PALIMPZEST_OPTIMIZER (pareto | none), PALIMPZEST_POLICY (MaxQuality | MinCost)
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import palimpzest_common as P  # noqa: E402

TAG = "palimpzest"
EXEC = os.path.join(HERE, "..", "palimpzest_hybrid_exec.py")


def main():
    which = sys.argv[1:] or [f"Q{i}" for i in range(1, 31)]
    out_dir = os.path.join(HERE, "results", TAG)
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    per_query = {}
    print(f"Palimpzest agent_bench {P.MODEL} via {P.PROXY}  optimizer={P.OPTIMIZER}  policy={P.POLICY}  workers={P.DOP}  queries={which}\n")
    for name in which:
        csv_path = os.path.join(out_dir, name + ".csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        r0, m0, p0, c0 = P.proxy_counters()
        t0 = time.time()
        cmd = [P.python(), EXEC, os.path.join(HERE, "lotus_queries", name + ".py"), os.path.join(HERE, "dataset"), csv_path,
               P.MODEL, P.PROXY, str(P.DOP), "--optimizer", P.OPTIMIZER, "--policy", P.POLICY]
        try:
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=7200, cwd=HERE)
        except subprocess.TimeoutExpired:  # Palimpzest has no cap of its own; record the overrun, keep going
            p = None
        dt = time.time() - t0  # wall time, interpreter start-up included; latency_s is the program's own time
        r1, m1, p1, c1 = P.proxy_counters()
        err, rows, secs, operators = None, 0, dt, []
        if p is None:
            err = "timeout after 7200 s"
        elif p.returncode != 0 or not p.stdout.strip():
            tail = [ln for ln in p.stderr.strip().splitlines() if ln.strip()]
            err = (tail[-1] if tail else f"exit {p.returncode}")[:300]
        else:
            res = json.loads(p.stdout)
            rows, secs, operators = res["rows"], res["seconds"], res["operators"]
        cost = round((p1 - p0) / 1e6 * P.P_IN + (c1 - c0) / 1e6 * P.P_OUT, 5)
        rec = {"latency_s": round(secs, 1), "wall_s": round(dt, 1), "llm_calls": r1 - r0, "llm_calls_fresh": m1 - m0,
               "tokens": (p1 - p0) + (c1 - c0), "cost_usd": cost, "cost_is_estimate": True, "optimizer": P.OPTIMIZER,
               "policy": P.POLICY, "rows": rows, "operators": operators, "error": err, "ran_at": stamp}
        per_query[name] = rec
        print(f"{name:>4} lat={secs:7.1f}s calls={rec['llm_calls']:>6} fresh={rec['llm_calls_fresh']:>6} tok={rec['tokens']:>8} ~${cost:.4f} rows={rows}"
              + (f"  ERR {err}" if err else ""), flush=True)
        path = os.path.join(HERE, "results", TAG + "_agentbench_results.json")
        old = json.load(open(path)) if os.path.exists(path) else {"per_query": {}}
        old["model"], old["concurrency"], old["optimizer"], old["policy"], old["updated"] = P.MODEL, P.DOP, P.OPTIMIZER, P.POLICY, stamp
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
