#!/usr/bin/env python3
"""Sema on the agent_bench 30 queries -- the authors' DuckDB fork (Sema, VLDB 2026; github.com/BITQiKangK/Sema at
a86cb62, built with aisql-bench/setup_sema.sh) on sema_queries/ (translate_to_sema.py: PLOP's queries in Sema's
s'...' dialect, prompt text verbatim), at Sema's default settings (a lone filter or map asks one row per request).

Every request passes through SWANBench's meter (copied into aisql-bench/swanbench_meter.py), which forwards it to
the cache proxy and counts it the way SWAN 2.0 counts every system's (requests incl. retries, tokens from usage,
cost from the replayed cost header). Sema's
answers are its own samples (its own prompts), like LOTUS's, BlendSQL's and Palimpzest's. Concurrency is the
other runners' 20 (SEMA_LLM_MAX_CONNECTIONS from sema_fork.patch, SET threads=1), with SEMA_LLM_TIMEOUT_S so a
queued request does not time out (sema_movie.py explains both). Result rows go to results/<tag>/QN.csv for
eval_agentbench.py.

The TPC-H tables come from the benchmark's parquet files (dataset/tpch/, which swan_queries/ reads): Sema's build
has neither the tpch nor the parquet extension, so SWAN-AISQL's shell reads the parquet once and Sema's own shell
writes dataset/tpch_sema.duckdb with the same column types; the queries attach it read-only.

  run:  python3 sema_agentbench.py [Q1 Q2 ...]
  env:  SEMA_BIN (default ../../../Sema/build/release/duckdb), SWAN_BIN (default ../../build/release/duckdb),
        AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna), AI_MAX_CONCURRENCY (20), SEMA_LLM_TIMEOUT_S (1800),
        SEMA_TAG (default sema)
A tag that already holds results takes more queries only from a run with the same model and request limits; any
other run stops before it writes anything (use another SEMA_TAG).
"""
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.dirname(HERE))
from swanbench_meter import Meter  # noqa: E402 -- aisql-bench/swanbench_meter.py, SWANBench's meter (stdlib only)

BIN = os.environ.get("SEMA_BIN", os.path.join(ROOT, "Sema", "build", "release", "duckdb"))
SWAN_BIN = os.environ.get("SWAN_BIN", os.path.join(HERE, "..", "..", "build", "release", "duckdb"))
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = os.environ.get("AI_MAX_CONCURRENCY", "20")
TIMEOUT_S = os.environ.get("SEMA_LLM_TIMEOUT_S", "1800")
TAG = os.environ.get("SEMA_TAG", "sema")
ENV = {**os.environ, "DUCKDB_AGENT_MODE": "0", "SEMA_LLM_MAX_CONNECTIONS": CONC, "SEMA_LLM_TIMEOUT_S": TIMEOUT_S}
CONFIG = {"model": MODEL, "workers": CONC, "request_timeout_s": TIMEOUT_S}  # Sema's own settings are its defaults
TPCH_TABLES = ("part", "supplier", "customer", "lineitem", "orders", "partsupp", "nation", "region")
TPCH_DB = os.path.join(HERE, "dataset", "tpch_sema.duckdb")
NULL = "\\N"


def _run(binary, script, *args):
    return subprocess.run([binary, *args], input=script, text=True, capture_output=True, env=ENV, cwd=HERE)


def ensure_tpch():
    """dataset/tpch_sema.duckdb, written under a temporary name and renamed once every table's row count matches its
    parquet file, so a failed or interrupted build never leaves a file that a later run would take as built."""
    if os.path.exists(TPCH_DB):
        return
    part = f"{TPCH_DB}.tmp{os.getpid()}"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            create, counts = [], {}
            for t in TPCH_TABLES:
                src = os.path.join(HERE, "dataset", "tpch", f"{t}.parquet")
                out = os.path.join(tmp, f"{t}.csv")
                p = _run(SWAN_BIN, f"DESCRIBE SELECT * FROM '{src}';\nSELECT count(*) FROM '{src}';\n"
                                   f"COPY (SELECT * FROM '{src}') TO '{out}' (HEADER, NULL '{NULL}', QUOTE '\"', ESCAPE '\"');\n",
                         "-csv", "-noheader")
                if p.returncode != 0:
                    raise SystemExit(f"reading {src} failed: {p.stderr.strip()[-300:]}")
                lines = list(csv.reader(io.StringIO(p.stdout)))
                cols = [(r[0], r[1]) for r in lines if len(r) >= 2 and r[1] and not r[0].isdigit()]
                counts[t] = int(next(r[0] for r in lines if len(r) == 1 and r[0].isdigit()))
                spec = ", ".join(f'"{c}" {ty}' for c, ty in cols)
                create.append(f'CREATE TABLE {t} ({spec});\n'
                              f'COPY {t} FROM \'{out}\' (HEADER, NULL \'{NULL}\', QUOTE \'"\', ESCAPE \'"\');')
            p = _run(BIN, "\n".join(create) + "\n", "-bail", part)
            if p.returncode != 0:
                raise SystemExit(f"writing {TPCH_DB} failed: {p.stderr.strip()[-300:]}")
        check = "\n".join(f"SELECT '{t}', count(*) FROM {t};" for t in TPCH_TABLES)
        p = _run(BIN, check, "-readonly", "-csv", "-noheader", part)
        got = {line.split(",")[0]: int(line.split(",")[1]) for line in p.stdout.split()}
        if got != counts:
            raise SystemExit(f"TPC-H copy lost rows: {counts} vs {got}")
        os.replace(part, TPCH_DB)
    finally:
        for f in (part, part + ".wal"):
            if os.path.exists(f):
                os.remove(f)


def proxy_replays_latency():
    """Whether the cache proxy returns recorded answers after their recorded response time (its /cache/stats
    `simulate_latency`); None when the endpoint does not say (not the cache proxy)."""
    try:
        with urllib.request.urlopen(PROXY.rstrip("/") + "/cache/stats", timeout=10) as r:
            return json.load(r).get("simulate_latency")
    except Exception:
        return None


def timing(fresh, replays_latency):
    """How a query's latency was measured: "live" when some of its calls were new to the cache (asked of the
    provider during the run), "replay" when the cache answered every call after its recorded response time."""
    if fresh is None:
        return None
    if fresh:
        return "live"
    return {True: "replay", False: "replay without recorded latency"}.get(replays_latency)


def incompatible(saved):
    """Why a saved run cannot take this run's queries (another model or request limits, or a batched run of an
    earlier version of this runner), or None."""
    for key, now in CONFIG.items():
        if key in saved and str(saved[key]) != str(now):
            return f"its {key} is {saved[key]}, this run's is {now}"
    if saved.get("batch", "default") != "default":
        return f"it batched {saved['batch']} rows per prompt, this run uses Sema's defaults"
    return None


def main():
    which = sys.argv[1:] or [f"Q{i}" for i in range(1, 31)]
    path = os.path.join(HERE, "results", TAG + "_agentbench_results.json")
    why = incompatible(json.load(open(path))) if os.path.exists(path) else None
    if why:
        raise SystemExit(f"results/{TAG} holds a run of another configuration ({why}): nothing written; use another "
                         f"SEMA_TAG, or move results/{TAG} and {os.path.basename(path)} away first")
    ensure_tpch()
    replays_latency = proxy_replays_latency()
    os.makedirs(os.path.join(HERE, "results", TAG), exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"SEMA agent_bench {MODEL} via meter -> {PROXY}  concurrency={CONC}  queries={which}", flush=True)
    with Meter(PROXY) as meter:
        for name in which:
            script = (".bail on\n"
                      f"SET llm_url='{meter.url}/v1/chat/completions';\nSET llm_model='{MODEL}';\n"
                      "SET llm_api_key='sk-test';\nSET threads=1;\n"
                      f".read sema_queries/{name}.sql\n")
            meter.reset()
            m0 = meter.upstream_misses()
            t0 = time.time()
            p = subprocess.run([BIN, "-csv"], input=script, env=ENV, text=True, capture_output=True, cwd=HERE,
                               timeout=int(os.environ.get("SEMA_QUERY_TIMEOUT_S", "43200")))
            dt = time.time() - t0
            use = meter.snapshot()
            m1 = meter.upstream_misses()
            err = None
            if p.returncode != 0 or "Error" in p.stderr:
                err = (p.stderr.strip().split("\n")[-1] or "nonzero exit")[:300]
            out = p.stdout.strip()
            rows = max(0, len(out.split("\n")) - 1) if out else 0
            fresh = (m1 - m0) if (m0 is not None and m1 is not None) else None
            rec = {"latency_s": round(dt, 1), "llm_calls": use["requests"], "errors": use["errors"],
                   "llm_calls_fresh": fresh, "timing": timing(fresh, replays_latency),
                   "tokens": use["prompt_tokens"] + use["completion_tokens"],
                   "reasoning_tokens": use["reasoning_tokens"], "cost_usd": round(use["cost_usd"], 5),
                   "rows": rows, "error": err, "ran_at": stamp}
            open(os.path.join(HERE, "results", TAG, name + ".csv"), "w").write(out + "\n")
            print(f"{name:>4} lat={dt:7.1f}s calls={rec['llm_calls']:>6} errors={rec['errors']:>4} "
                  f"fresh={str(rec['llm_calls_fresh']):>6} tok={rec['tokens']:>8} ${rec['cost_usd']:.4f} rows={rows}"
                  + (f"  ERR {err}" if err else ""), flush=True)
            old = json.load(open(path)) if os.path.exists(path) else {"per_query": {}}
            old.update({**CONFIG, "updated": stamp})
            old["per_query"][name] = rec
            tot = old["per_query"]
            old["total"] = {"n_queries": len(tot),
                            "latency_s": round(sum(q["latency_s"] for q in tot.values()), 1),
                            "llm_calls": sum(q["llm_calls"] for q in tot.values()),
                            "cost_usd": round(sum(q["cost_usd"] for q in tot.values()), 4),
                            "tokens": sum(q["tokens"] for q in tot.values()),
                            "errors": sum(1 for q in tot.values() if q["error"])}
            json.dump(old, open(path, "w"), indent=1)
    print("\nTOTAL", json.load(open(path))["total"])


if __name__ == "__main__":
    main()
