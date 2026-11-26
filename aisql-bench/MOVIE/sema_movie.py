#!/usr/bin/env python3
"""Sema on SemBench MOVIE sf_2000 -- the authors' DuckDB fork (Sema, VLDB 2026; github.com/BITQiKangK/Sema at
a86cb62, built with aisql-bench/sema_fork.patch) on the ten queries written in its dialect (sema_queries/qN.sql:
s'...' semantic expressions), with the same data, model, proxy and metrics (movie_common) as swan_movie.py,
lotus_movie.py and plop_movie.py.

Sema calls an OpenAI-compatible chat-completions endpoint (its llm_url / llm_model / llm_api_key settings) with
its own prompts, so its answers are independent samples. Every request passes through SWANBench's meter (copied
into aisql-bench/swanbench_meter.py), which forwards it to the cache proxy and counts it the way SWAN 2.0 counts
every system's: requests (retries included),
tokens from the response's usage, and cost from the cost header the proxy replays (litellm's gpt-5.6-luna
prices). Sema runs with its own settings: a lone filter or map asks one row per request (semantic_batch_size 1).
SEMA_BATCH=k sets semantic_batch_size, and semantic_filter_batch_size for its multi-filter executor, to k rows per
prompt; SEMA_EXTRA_SET adds other SET statements.

Concurrency is the other runners' 20: sema_fork.patch reads SEMA_LLM_MAX_CONNECTIONS (requests in flight per
thread; Sema's own limit is 50) and SEMA_LLM_TIMEOUT_S (Sema's 30 s request timeout runs from the moment a request
is queued, so with a cap the queued tail of a 2,048-row chunk would time out and be retried), and every query runs
with SET threads=1 so the cap holds for the process. The relational work here is a few thousand rows.

  run (from aisql-bench/MOVIE):  python3 sema_movie.py [q1 q2 ...]
  env:  SEMA_BIN (default ../../../Sema/build/release/duckdb), AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna),
        AI_MAX_CONCURRENCY (20), SEMA_LLM_TIMEOUT_S (1800), SEMA_BATCH (unset: Sema's own), SEMA_EXTRA_SET,
        SWAN_TAG (suffix of the result files)
"""
import csv
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from movie_common import gt, metric_fields, metric_tag, score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.dirname(HERE))
from swanbench_meter import Meter  # noqa: E402 -- aisql-bench/swanbench_meter.py, SWANBench's meter (stdlib only)

BIN = os.environ.get("SEMA_BIN", os.path.join(ROOT, "Sema", "build", "release", "duckdb"))
DB = os.path.join(HERE, "movie_sema.db")  # Sema (a DuckDB 1.2-era fork) cannot open movie.db (DuckDB 2.0 storage)
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
TAG = os.environ.get("SWAN_TAG", "")
CONC = os.environ.get("AI_MAX_CONCURRENCY", "20")
TIMEOUT_S = os.environ.get("SEMA_LLM_TIMEOUT_S", "1800")
BATCH = os.environ.get("SEMA_BATCH", "")
EXTRA = ("SET threads=1;\n"
         + (f"SET semantic_batch_size={BATCH};\nSET semantic_filter_batch_size={BATCH};\n" if BATCH else "")
         + os.environ.get("SEMA_EXTRA_SET", ""))
ENV = {**os.environ, "DUCKDB_AGENT_MODE": "0", "SEMA_LLM_MAX_CONNECTIONS": CONC, "SEMA_LLM_TIMEOUT_S": TIMEOUT_S}
SETTINGS = ("semantic_batch_size", "semantic_voting_rounds", "semantic_filter_batch_size",
            "semantic_filter_voting_rounds", "semantic_filter_accuracy_threshold", "semantic_filter_latency_first",
            "threads")


def sema(script, db=DB, readonly=True):
    args = [BIN] + (["-readonly"] if readonly else []) + ["-csv", db]
    return subprocess.run(args, input=script, text=True, capture_output=True, cwd=HERE, env=ENV,
                          timeout=int(os.environ.get("SEMA_QUERY_TIMEOUT_S", "43200")))


def ensure_db():
    """movie_sema.db from setup.sql, written by Sema's shell under a temporary name and renamed once both tables are
    in it, so a failed or interrupted build never leaves a file that a later run would take as built."""
    if os.path.exists(DB):
        return
    part = f"{DB}.tmp{os.getpid()}"
    try:
        p = sema(".bail on\n" + open(os.path.join(HERE, "setup.sql")).read(), db=part, readonly=False)
        if p.returncode != 0:
            raise SystemExit("could not build movie_sema.db: " + p.stderr.strip())
        p = sema("SELECT (SELECT count(*) FROM movies) > 0 AND (SELECT count(*) FROM reviews) > 0;\n", db=part)
        if p.returncode != 0 or "true" not in p.stdout:
            raise SystemExit("could not build movie_sema.db: the movies or reviews table is missing or empty")
        os.replace(part, DB)
    finally:
        for f in (part, part + ".wal"):
            if os.path.exists(f):
                os.remove(f)


def effective_settings():
    q = "SELECT " + ", ".join(f"current_setting('{s}') AS {s}" for s in SETTINGS) + ";"
    p = sema(".bail on\n" + EXTRA + "\n" + q + "\n")
    rows = list(csv.DictReader(io.StringIO(p.stdout)))
    return rows[0] if rows else {"error": p.stderr.strip()[-200:]}


def run_one(name, meter, stamp):
    script = (".bail on\n"
              f"SET llm_url='{meter.url}/v1/chat/completions';\n"
              f"SET llm_model='{MODEL}';\n"
              "SET llm_api_key='sk-test';\n"
              f"{EXTRA}\n"
              f".read sema_queries/{name}.sql\n")
    meter.reset()
    m0 = meter.upstream_misses()
    t0 = time.time()
    p = sema(script)
    dt = time.time() - t0
    use = meter.snapshot()
    m1 = meter.upstream_misses()
    rows = list(csv.reader(io.StringIO(p.stdout)))
    out = [tuple(r) for r in rows[1:]]  # rows[0] is the header
    err = None
    if p.returncode != 0:
        last = (p.stderr.strip().splitlines() or ["(no message)"])[-1]
        err = "sql: " + last[:200]
    gold = gt(name)
    P, R, M = score(name, out, gold) if err is None else (0.0, 0.0, 0.0)
    rec = {"latency_s": round(dt, 1),
           "llm_calls": use["requests"], "errors": use["errors"],
           "llm_calls_fresh": (m1 - m0) if (m0 is not None and m1 is not None) else None,
           "prompt_tokens": use["prompt_tokens"], "completion_tokens": use["completion_tokens"],
           "reasoning_tokens": use["reasoning_tokens"],
           "tokens": use["prompt_tokens"] + use["completion_tokens"], "cost_usd": round(use["cost_usd"], 5),
           "n_out": len(out), "n_gold": len(gold), **metric_fields(name, P, R),
           "quality": M, "error": err, "ran_at": stamp}
    raw_entry = {"predicted": [list(x) for x in out][:5000]}
    tag = f"ERR {err}" if err else f"{metric_tag(name, P, R)} quality={M}"
    line = (f"{name:4} lat={rec['latency_s']:7.1f}s calls={rec['llm_calls']:>6} errors={rec['errors']:>4} "
            f"fresh={str(rec['llm_calls_fresh']):>6} tok={rec['tokens']:>9} ${rec['cost_usd']:.4f}  "
            f"n={rec['n_out']}/{rec['n_gold']}  {tag}")
    return rec, raw_entry, line


def main():
    which = sys.argv[1:] or [f"q{i}" for i in range(1, 11)]
    res_p, raw_p, log_p = (os.path.join(HERE, f) for f in
                           (f"sema_movie_results{TAG}.json", f"sema_movie_raw{TAG}.json", f"sema_movie_log{TAG}.txt"))

    def _load(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}

    per_query = _load(res_p).get("per_query", {})
    raw = _load(raw_p)
    ensure_db()
    settings = effective_settings()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    settings = {**settings, "max_connections": CONC, "request_timeout_s": TIMEOUT_S}
    header = f"SEMA MOVIE {MODEL} via meter -> {PROXY}  sf2000  serial  settings={settings}  queries={which}"
    print(header + "\n", flush=True)
    lines = []
    with Meter(PROXY) as meter:
        for name in which:
            rec, raw_entry, line = run_one(name, meter, stamp)
            per_query[name], raw[name] = rec, raw_entry
            print(line, flush=True)
            lines.append(line)
    allq = per_query
    tot = {"n_queries": len(allq),
           "latency_s": round(sum(r["latency_s"] for r in allq.values()), 1),
           "llm_calls": sum(r["llm_calls"] for r in allq.values()),
           "errors": sum(r["errors"] for r in allq.values()),
           "cost_usd": round(sum(r["cost_usd"] for r in allq.values()), 4),
           "tokens": sum(r["tokens"] for r in allq.values()),
           "macro_quality": round(sum(r["quality"] for r in allq.values()) / len(allq), 3)}
    tline = (f"TOTAL({len(allq)}q)  latency={tot['latency_s']}s  calls={tot['llm_calls']}  errors={tot['errors']}  "
             f"cost=${tot['cost_usd']}  tokens={tot['tokens']}  macro-quality={tot['macro_quality']}")
    print("\n" + tline)
    json.dump({"model": MODEL, "settings": settings, "extra_set": EXTRA, "updated": stamp,
               "per_query": per_query, "total": tot}, open(res_p, "w"), indent=1)
    json.dump(raw, open(raw_p, "w"), indent=1)
    with open(log_p, "a") as f:
        f.write(f"\n== {stamp} ==\n{header}\n" + "\n".join(lines) + "\n" + tline + "\n")
    print(f"logged -> {os.path.basename(res_p)}")


if __name__ == "__main__":
    main()
