#!/usr/bin/env python3
"""PLOP (Morrila) on its own agent_bench 30 queries, DP cost-model mode (the paper's setting).

Runs the authors' DuckDB fork on their unmodified QN.sql files with
SEMANTIC_OPTIMIZER=true, DUCKDB_SEMANTIC_MODE=costmodel, workers pinned to our concurrency,
model gpt-5.6-luna, routed via the :4001 recording proxy (LLM_API_URL patch). True call
counting = proxy hits+misses delta; tokens from the proxy's prompt/completion counters; cost
ESTIMATED from tokens at luna list price (their /v1/responses path bypasses litellm's cost
header on replay). Result rows go to results/plop/QN.csv for cross-system agreement.

  run:  python3 plop_agentbench.py [Q1 Q2 ...]
  env:  PLOP_ROOT (the morrila_src checkout), AI_PROXY_URL, AI_MODEL, AI_MAX_CONCURRENCY
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get(
    "PLOP_ROOT",
    "/private/tmp/claude-501/-Users-owner-Desktop-research-swan-ai-sql/9711bdca-3e49-49d5-aa05-4d5b2fb63486/scratchpad/morrila_src")
BIN = os.path.join(ROOT, "build", "reldebug", "duckdb")
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = os.environ.get("AI_MAX_CONCURRENCY", "20")
# gpt-5.6 list price per 1M tokens (input / output); estimate only
P_IN, P_OUT = 0.25, 2.00

# which query folder inside the PLOP tree to read (agent_bench_gt = output LIMITs removed)
QUERY_DIR = os.environ.get("PLOP_QUERY_DIR", "test/semantic/agent_bench")
TAG_OVERRIDE = os.environ.get("PLOP_TAG", "")
MODE = os.environ.get("PLOP_MODE", "costmodel")  # costmodel = the paper's DP; none = the paper's QUALITY GROUND TRUTH (un-rewritten UDF execution)
ENV = {**os.environ, "SEMANTIC_OPTIMIZER": "true", "DUCKDB_SEMANTIC_MODE": MODE,
       "LLM_API_URL": PROXY + "/v1/responses", "API_KEY": "sk-test", "LLM_MODEL": MODEL,
       "NUM_LLM_WORKERS": CONC}


def stats():
    try:
        s = json.load(urllib.request.urlopen(PROXY + "/cache/stats", timeout=5))
        return s["hits"] + s["misses"], s["misses"], s.get("prompt_tokens", 0), s.get("completion_tokens", 0)
    except Exception:
        return 0, -1, 0, 0


def main():
    which = sys.argv[1:] or [f"Q{i}" for i in range(1, 31)]
    tag = TAG_OVERRIDE or ("plop" if MODE == "costmodel" else f"plop_{MODE}")
    os.makedirs(os.path.join(HERE, "results", tag), exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    per_query = {}
    log_path = os.path.join(ROOT, f"llm_calls_{MODEL}.log")
    for name in which:
        log0 = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        r0, m0, pt0, ct0 = stats()
        t0 = time.time()
        p = subprocess.run([BIN, "-csv", "-c", f".read {QUERY_DIR}/{name}.sql"],
                           env=ENV, text=True, capture_output=True, cwd=ROOT, timeout=3000)
        dt = time.time() - t0
        r1, m1, pt1, ct1 = stats()
        # tokens from PLOP's own per-process log (responses-API usage; proxy counters do not parse it)
        p_tok = c_tok = 0
        try:
            with open(log_path) as fh:
                fh.seek(log0)
                chunk = fh.read()
            import re as _re
            totals = _re.findall(r"Total tokens so far: \d+ \(P: (\d+), C: (\d+)", chunk)
            if totals:
                p_tok, c_tok = int(totals[-1][0]), int(totals[-1][1])
        except Exception:
            pass
        err = None
        if p.returncode != 0 or "Error" in p.stderr:
            err = (p.stderr.strip().split("\n")[-1] or "nonzero exit")[:300]
        out = p.stdout.strip()
        rows = max(0, len(out.split("\n")) - 1) if out else 0
        cost = round(p_tok / 1e6 * P_IN + c_tok / 1e6 * P_OUT, 5)
        rec = {"latency_s": round(dt, 1), "llm_calls": r1 - r0, "llm_calls_fresh": m1 - m0,
               "tokens": p_tok + c_tok, "cost_usd_est": cost, "rows": rows,
               "error": err, "ran_at": stamp}
        per_query[name] = rec
        open(os.path.join(HERE, "results", tag, name + ".csv"), "w").write(out + "\n")
        print(f"{name:>4} lat={dt:7.1f}s calls={rec['llm_calls']:>6} fresh={rec['llm_calls_fresh']:>6} "
              f"tok={rec['tokens']:>8} ~${cost:.4f} rows={rows}" + (f"  ERR {err}" if err else ""), flush=True)
        path = os.path.join(HERE, "results", tag + "_agentbench_results.json")
        old = json.load(open(path)) if os.path.exists(path) else {"per_query": {}}
        old["model"], old["mode"], old["workers"], old["updated"] = MODEL, MODE, CONC, stamp
        old["per_query"].update(per_query)
        tot = old["per_query"]
        old["total"] = {"n_queries": len(tot),
                        "latency_s": round(sum(q["latency_s"] for q in tot.values()), 1),
                        "llm_calls": sum(q["llm_calls"] for q in tot.values()),
                        "cost_usd_est": round(sum(q["cost_usd_est"] for q in tot.values()), 4),
                        "tokens": sum(q["tokens"] for q in tot.values()),
                        "errors": sum(1 for q in tot.values() if q["error"])}
        json.dump(old, open(path, "w"), indent=1)
    print("\nTOTAL", json.load(open(os.path.join(HERE, "results", tag + "_agentbench_results.json")))["total"])


main()
