#!/usr/bin/env python3
"""PLOP (Morrila) on SemBench MOVIE sf_2000 -- the authors' DuckDB fork in its DP cost-model mode, on the
ten queries written in its dialect (plop_queries/qN.sql: semantic / semantic_int / semantic_string), the same
data, model, proxy and metrics (movie_common) as swan_movie.py and lotus_movie.py.

PLOP is text-only, so this is the one SemBench suite it runs in full. Its prompts carry its own answer-format
suffix, so its verdicts are independent samples (like LOTUS's), not the shared samples of the hybrid bench.
Calls = proxy hits+misses delta per query; tokens from PLOP's own log (its /v1/responses path carries no cost
header through the proxy), cost ESTIMATED at the luna list price used by the hybrid-bench runner.

  run:  python3 plop_movie.py [q1 q2 ...]
  env:  PLOP_BIN (the fork's duckdb shell, built with the LLM_API_URL / LLM_MODEL patch), AI_PROXY_URL (:4001),
        AI_MODEL (gpt-5.6-luna), AI_MAX_CONCURRENCY (20), PLOP_MODE (costmodel | pullup | none)
"""
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

from movie_common import gt, metric_fields, metric_tag, score

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("PLOP_BIN", os.path.join(HERE, "..", "..", "..", "MorrilaPLOP", "PLOP", "build", "release", "duckdb"))
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = os.environ.get("AI_MAX_CONCURRENCY", "20")
MODE = os.environ.get("PLOP_MODE", "costmodel")
TAG = os.environ.get("SWAN_TAG", "")
P_IN, P_OUT = 0.25, 2.00  # gpt-5.6 list price per 1M tokens (input / output); estimate only
ENV = {**os.environ, "SEMANTIC_OPTIMIZER": "true", "DUCKDB_SEMANTIC_MODE": MODE,
       "LLM_API_URL": PROXY + "/v1/responses", "API_KEY": "sk-test", "LLM_MODEL": MODEL, "NUM_LLM_WORKERS": CONC}
LOG_PATH = os.path.join(HERE, f"llm_calls_{MODEL}.log")


def stats():
    try:
        s = json.load(urllib.request.urlopen(PROXY + "/cache/stats", timeout=5))
        return s["hits"] + s["misses"], s["misses"]
    except Exception:
        return 0, -1


def run_one(name, stamp):
    log0 = os.path.getsize(LOG_PATH) if os.path.exists(LOG_PATH) else 0
    r0, m0 = stats()
    t0 = time.time()
    p = subprocess.run([BIN, "-csv", "-c", f".read plop_queries/{name}.sql"], env=ENV, text=True,
                       capture_output=True, cwd=HERE, timeout=7200)
    dt = time.time() - t0
    r1, m1 = stats()
    p_tok = c_tok = 0
    n_prompts = None
    try:
        with open(LOG_PATH) as fh:
            fh.seek(log0)
            chunk = fh.read()
        totals = re.findall(r"Total tokens so far: \d+ \(P: (\d+), C: (\d+)", chunk)
        if totals:
            p_tok, c_tok = int(totals[-1][0]), int(totals[-1][1])
        n_prompts = chunk.count("| PROMPT:")  # PLOP's own count of requests: exact even when other runs share the proxy
    except Exception:
        pass
    err = None
    if p.returncode != 0 or "Error" in p.stderr:
        err = "sql: " + (p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "nonzero exit")[:200]
    rows = list(csv.reader(io.StringIO(p.stdout))) if p.stdout.strip() else []
    out = [tuple(r) for r in rows[1:]]  # the shell's CSV mode prints a header row
    gold = gt(name)
    P, R, M = score(name, out, gold) if err is None else (0.0, 0.0, 0.0)
    cost = round(p_tok / 1e6 * P_IN + c_tok / 1e6 * P_OUT, 5)
    rec = {"latency_s": round(dt, 1), "llm_calls_fresh": m1 - m0, "llm_calls": n_prompts if n_prompts is not None else r1 - r0,
           "cache_hits": None,
           "tokens": p_tok + c_tok, "reasoning_tokens": None, "cost_usd": cost, "cost_is_estimate": True,
           "n_out": len(out), "n_gold": len(gold), **metric_fields(name, P, R),
           "quality": M, "error": err, "ran_at": stamp}
    raw_entry = {"predicted": [list(x) for x in out][:5000]}
    tag = f"ERR {err}" if err else f"{metric_tag(name, P, R)} quality={M}"
    line = (f"{name:4} lat={rec['latency_s']:7.1f}s calls={rec['llm_calls']:>6} fresh={rec['llm_calls_fresh']:>5} "
            f"tok={rec['tokens']:>9} ~${cost:.4f}  n={rec['n_out']}/{rec['n_gold']}  {tag}")
    return rec, raw_entry, line


def main():
    which = sys.argv[1:] or [f"q{i}" for i in range(1, 11)]
    RES, RAW, LOG = (os.path.join(HERE, f) for f in
                     (f"plop_movie_results{TAG}.json", f"plop_movie_raw{TAG}.json", f"plop_movie_log{TAG}.txt"))

    def _load(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    per_query = _load(RES).get("per_query", {})
    raw = _load(RAW)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header = f"PLOP MOVIE {MODEL} via {PROXY}  mode={MODE}  workers={CONC}  sf2000  serial  queries={which}"
    print(header + "\n")
    lines = []
    for name in which:  # serial: PLOP's worker pool is the concurrency; one query at a time keeps the counting exact
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
    json.dump({"model": MODEL, "mode": MODE, "concurrency": CONC, "updated": stamp, "per_query": per_query, "total": tot},
              open(RES, "w"), indent=1)
    json.dump(raw, open(RAW, "w"), indent=1)
    with open(LOG, "a") as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + tline + "\n\n")


if __name__ == "__main__":
    main()
