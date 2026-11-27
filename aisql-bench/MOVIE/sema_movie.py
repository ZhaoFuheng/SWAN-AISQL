#!/usr/bin/env python3
"""Sema on SemBench MOVIE sf_2000 -- the authors' DuckDB fork (Sema, VLDB 2026; github.com/BITQiKangK/Sema at
a86cb62, built with aisql-bench/setup_sema.sh) on the ten queries written in its dialect (sema_queries/qN.sql:
s'...' semantic expressions), with the same data, model, proxy and metrics (movie_common) as swan_movie.py,
lotus_movie.py and plop_movie.py.

Sema calls an OpenAI-compatible chat-completions endpoint (its llm_url / llm_model / llm_api_key settings) with
its own prompts, so its answers are independent samples. Every request passes through SWANBench's meter (copied
into aisql-bench/swanbench_meter.py), which forwards it to the cache proxy and counts it the way SWAN 2.0 counts
every system's: requests (retries included), tokens from the response's usage, and cost from the cost header the
proxy replays (litellm's gpt-5.6-luna prices). Sema runs at its default settings, where a lone filter or map asks
one row per request (semantic_batch_size 1).

Concurrency is the other runners' 20: sema_fork.patch reads SEMA_LLM_MAX_CONNECTIONS (requests in flight per
thread; Sema's own limit is 50) and SEMA_LLM_TIMEOUT_S (Sema's 30 s request timeout runs from the moment a request
is queued, so with a cap the queued tail of a 2,048-row chunk would time out and be retried), and every query runs
with SET threads=1 so the cap holds for the process. The relational work here is a few thousand rows.

  run (from aisql-bench/MOVIE):  python3 sema_movie.py [q1 q2 ...]
  env:  SEMA_BIN (default ../../../Sema/build/release/duckdb), AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna),
        AI_MAX_CONCURRENCY (20), SEMA_LLM_TIMEOUT_S (1800), SEMA_QUERY_TIMEOUT_S (43200, after which a query is
        recorded as failed), SWAN_TAG (suffix of the result files)

The run stops before it writes anything unless the binary is the patched build at Sema's shipped settings
(sema_common.py), and unless the results file is empty or holds a run of the same model, request limits and
settings: a file is one configuration (use another SWAN_TAG for another). A query during which a model request
failed is recorded as failed, since Sema fills in false or an empty string where an answer is missing.
"""
import csv
import io
import json
import os
import sys
from datetime import datetime, timezone

from movie_common import gt, metric_fields, metric_tag, score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sema_common as S  # noqa: E402 -- aisql-bench/sema_common.py

DB = os.path.join(HERE, "movie_sema.db")  # Sema (a DuckDB 1.2-era fork) cannot open movie.db (DuckDB 2.0 storage)
TAG = os.environ.get("SWAN_TAG", "")


def ensure_db():
    """movie_sema.db from setup.sql, written by Sema's shell under a temporary name and renamed once both tables are
    in it, so a failed or interrupted build never leaves a file that a later run would take as built."""
    if os.path.exists(DB):
        return
    part = f"{DB}.tmp{os.getpid()}"
    try:
        p = S.shell(".bail on\n" + open(os.path.join(HERE, "setup.sql")).read() + "\nCHECKPOINT;\n", "-csv", part, cwd=HERE)
        if p.returncode != 0:
            raise SystemExit("could not build movie_sema.db: " + p.stderr.strip())
        p = S.shell("SELECT (SELECT count(*) FROM movies) > 0 AND (SELECT count(*) FROM reviews) > 0;\n",
                    "-readonly", "-csv", part, cwd=HERE)
        if p.returncode != 0 or "true" not in p.stdout:
            raise SystemExit("could not build movie_sema.db: the movies or reviews table is missing or empty")
        os.replace(part, DB)
    finally:
        for f in (part, part + ".wal"):
            if os.path.exists(f):
                os.remove(f)


def run_one(name, meter, stamp, replays_latency):
    query = os.path.join(HERE, "sema_queries", name + ".sql")
    meter.reset()
    m0 = meter.upstream_misses()
    p, dt = S.run_query(S.session(meter.url) + f".read sema_queries/{name}.sql\n", "-readonly", "-csv", DB, cwd=HERE)
    use = meter.snapshot()
    m1 = meter.upstream_misses()
    err = S.query_error(p, use)
    rows = list(csv.reader(io.StringIO(p.stdout))) if p is not None else []
    out = [tuple(r) for r in rows[1:]]  # rows[0] is the header
    gold = gt(name)
    P, R, M = score(name, out, gold) if err is None else (0.0, 0.0, 0.0)
    fresh = (m1 - m0) if (m0 is not None and m1 is not None) else None
    rec = {"latency_s": round(dt, 1),
           "llm_calls": use["requests"], "errors": use["errors"], "unanswered": use["unanswered"],
           "llm_calls_fresh": fresh, "timing": S.timing(fresh, replays_latency),
           "prompt_tokens": use["prompt_tokens"], "completion_tokens": use["completion_tokens"],
           "reasoning_tokens": use["reasoning_tokens"],
           "tokens": use["prompt_tokens"] + use["completion_tokens"], "cost_usd": round(use["cost_usd"], 5),
           "n_out": len(out), "n_gold": len(gold), **metric_fields(name, P, R),
           "quality": M, "query_sha256": S.sha256_of(query), "error": err, "ran_at": stamp}
    raw_entry = {"predicted": [list(x) for x in out][:5000]}
    tag = f"ERR {err}" if err else f"{metric_tag(name, P, R)} quality={M}"
    line = (f"{name:4} lat={rec['latency_s']:7.1f}s calls={rec['llm_calls']:>6} errors={rec['errors']:>4} "
            f"fresh={str(rec['llm_calls_fresh']):>6} tok={rec['tokens']:>9} ${rec['cost_usd']:.4f}  "
            f"n={rec['n_out']}/{rec['n_gold']}  {tag}")
    return rec, raw_entry, line


def totals(per_query):
    return {"n_queries": len(per_query),
            "latency_s": round(sum(r["latency_s"] for r in per_query.values()), 1),
            "llm_calls": sum(r["llm_calls"] for r in per_query.values()),
            "errors": sum(r["errors"] for r in per_query.values()),
            "unanswered_requests": sum(r["unanswered"] for r in per_query.values()),
            "failed_queries": sum(1 for r in per_query.values() if r["error"]),
            "cost_usd": round(sum(r["cost_usd"] for r in per_query.values()), 4),
            "tokens": sum(r["tokens"] for r in per_query.values()),
            "macro_quality": round(sum(r["quality"] for r in per_query.values()) / len(per_query), 3)}


def main():
    which = sys.argv[1:] or [f"q{i}" for i in range(1, 11)]
    res_p, raw_p, log_p = (os.path.join(HERE, f) for f in
                           (f"sema_movie_results{TAG}.json", f"sema_movie_raw{TAG}.json", f"sema_movie_log{TAG}.txt"))
    saved = json.load(open(res_p)) if os.path.exists(res_p) else {}
    why = S.incompatible(saved, S.configuration())  # model and request limits first: they need no Sema to compare
    config = S.configuration(S.check_build(HERE)) if not why else None
    why = why or S.incompatible(saved, config)
    if why:
        raise SystemExit(f"{os.path.basename(res_p)} holds a run of another configuration ({why}): nothing written; "
                         f"use another SWAN_TAG, or move the sema_movie_*{TAG} files away first")
    missing = [q for q in which if not os.path.exists(os.path.join(HERE, "sema_queries", q + ".sql"))]
    if missing:
        raise SystemExit(f"no such query: {', '.join(missing)} (sema_queries/q1.sql .. q10.sql)")
    per_query = saved.get("per_query", {})
    raw = json.load(open(raw_p)) if os.path.exists(raw_p) and per_query else {}
    ensure_db()
    replays_latency = S.proxy_replays_latency()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header = (f"SEMA MOVIE {S.MODEL} via meter -> {S.PROXY}  sf2000  serial  concurrency={S.CONC}  "
              f"settings={config['settings']}  queries={which}")
    print(header + "\n", flush=True)
    lines = []
    with S.Meter(S.PROXY) as meter:
        for name in which:
            rec, raw_entry, line = run_one(name, meter, stamp, replays_latency)
            per_query[name], raw[name] = rec, raw_entry
            print(line, flush=True)
            lines.append(line)
            # after every query, so a later failure or timeout keeps the queries that finished
            S.write(res_p, json.dumps({**config, "updated": stamp, "per_query": per_query, "total": totals(per_query)},
                                      indent=1))
            S.write(raw_p, json.dumps(raw, indent=1))
    tot = totals(per_query)
    tline = (f"TOTAL({len(per_query)}q)  latency={tot['latency_s']}s  calls={tot['llm_calls']}  errors={tot['errors']}  "
             f"cost=${tot['cost_usd']}  tokens={tot['tokens']}  macro-quality={tot['macro_quality']}")
    print("\n" + tline)
    with open(log_p, "a") as f:
        f.write(f"\n== {stamp} ==\n{header}\n" + "\n".join(lines) + "\n" + tline + "\n")
    print(f"logged -> {os.path.basename(res_p)}")


if __name__ == "__main__":
    main()
