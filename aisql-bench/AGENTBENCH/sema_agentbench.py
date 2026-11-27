#!/usr/bin/env python3
"""Sema on the agent_bench 30 queries -- the authors' DuckDB fork (Sema, VLDB 2026; github.com/BITQiKangK/Sema at
a86cb62, built with aisql-bench/setup_sema.sh) on sema_queries/ (translate_to_sema.py: PLOP's queries in Sema's
s'...' dialect), at Sema's default settings (a lone filter or map asks one row per request).

Every request passes through SWANBench's meter (copied into aisql-bench/swanbench_meter.py), which forwards it to
the cache proxy and counts it the way SWAN 2.0 counts every system's (requests incl. retries, tokens from usage,
cost from the replayed cost header). Sema's answers are its own samples (its own prompts), like LOTUS's, BlendSQL's
and Palimpzest's. Concurrency is the other runners' 20 (SEMA_LLM_MAX_CONNECTIONS from sema_fork.patch, SET
threads=1), with SEMA_LLM_TIMEOUT_S so a queued request does not time out (sema_movie.py explains both). Result
rows go to results/<tag>/QN.csv for eval_agentbench.py (an output over 20 MB as QN.csv.gz, which the scorer reads).

The TPC-H tables come from the benchmark's parquet files (dataset/tpch/, which swan_queries/ reads): Sema's build
has neither the tpch nor the parquet extension, so SWAN-AISQL's shell reads the parquet once and Sema's own shell
writes dataset/tpch_sema.duckdb with the same column types; the queries attach it read-only.

  run:  python3 sema_agentbench.py [Q1 Q2 ...]
  env:  SEMA_BIN (default ../../../Sema/build/release/duckdb), DUCKDB_BIN (SWAN-AISQL's shell, default
        ../../build/release/duckdb or reldebug; only to read the parquet files once), AI_PROXY_URL (:4001),
        AI_MODEL (gpt-5.6-luna), AI_MAX_CONCURRENCY (20), SEMA_LLM_TIMEOUT_S (1800), SEMA_QUERY_TIMEOUT_S (43200,
        after which a query is recorded as failed), SEMA_TAG (default sema)

The run stops before it writes anything unless the binary is the patched build at Sema's shipped settings
(sema_common.py), and unless the tag is empty or holds a run of the same model, request limits and settings: a tag
is one configuration (use another SEMA_TAG for another). A query during which a model request failed is recorded
as failed, since Sema fills in false or an empty string where an answer is missing.
"""
import csv
import gzip
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sema_common as S  # noqa: E402 -- aisql-bench/sema_common.py

_BUILDS = [os.path.abspath(os.path.join(HERE, "../../build", b, "duckdb")) for b in ("release", "reldebug")]
SWAN_BIN = os.environ.get("DUCKDB_BIN", next((p for p in _BUILDS if os.path.exists(p)), _BUILDS[0]))
TAG = os.environ.get("SEMA_TAG", "sema")
TPCH_TABLES = ("part", "supplier", "customer", "lineitem", "orders", "partsupp", "nation", "region")
TPCH_DB = os.path.join(HERE, "dataset", "tpch_sema.duckdb")
NULL = "\\N"
GZIP_OVER = 20 * 1024 * 1024  # bytes: a larger output is kept gzipped, the form the big outputs are committed in


def ensure_tpch():
    """dataset/tpch_sema.duckdb, written under a temporary name and renamed once every table's row count matches its
    parquet file, so a failed or interrupted build never leaves a file that a later run would take as built."""
    if os.path.exists(TPCH_DB):
        return
    if not os.path.isfile(SWAN_BIN):
        raise SystemExit(f"no SWAN-AISQL shell at {SWAN_BIN} to read dataset/tpch/*.parquet with: build it, or set DUCKDB_BIN")
    part = f"{TPCH_DB}.tmp{os.getpid()}"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            create, counts = [], {}
            for t in TPCH_TABLES:
                src = os.path.join(HERE, "dataset", "tpch", f"{t}.parquet")
                out = os.path.join(tmp, f"{t}.csv")
                p = S.shell(f"DESCRIBE SELECT * FROM '{src}';\nSELECT count(*) FROM '{src}';\n"
                            f"COPY (SELECT * FROM '{src}') TO '{out}' (HEADER, NULL '{NULL}', QUOTE '\"', ESCAPE '\"');\n",
                            "-csv", "-noheader", cwd=HERE, binary=SWAN_BIN)
                if p.returncode != 0:
                    raise SystemExit(f"reading {src} failed: {p.stderr.strip()[-300:]}")
                lines = list(csv.reader(io.StringIO(p.stdout)))
                cols = [(r[0], r[1]) for r in lines if len(r) >= 2 and r[1] and not r[0].isdigit()]
                counts[t] = int(next(r[0] for r in lines if len(r) == 1 and r[0].isdigit()))
                spec = ", ".join(f'"{c}" {ty}' for c, ty in cols)
                create.append(f'CREATE TABLE {t} ({spec});\n'
                              f'COPY {t} FROM \'{out}\' (HEADER, NULL \'{NULL}\', QUOTE \'"\', ESCAPE \'"\');')
            p = S.shell("\n".join(create) + "\nCHECKPOINT;\n", "-bail", part, cwd=HERE)
            if p.returncode != 0:
                raise SystemExit(f"writing {TPCH_DB} failed: {p.stderr.strip()[-300:]}")
        check = "\n".join(f"SELECT '{t}', count(*) FROM {t};" for t in TPCH_TABLES)
        p = S.shell(check, "-readonly", "-csv", "-noheader", part, cwd=HERE)
        got = {line.split(",")[0]: int(line.split(",")[1]) for line in p.stdout.split()}
        if got != counts:
            raise SystemExit(f"TPC-H copy lost rows: {counts} vs {got}")
        os.replace(part, TPCH_DB)
    finally:
        for f in (part, part + ".wal"):
            if os.path.exists(f):
                os.remove(f)


def write_rows(path, text):
    """A query's rows at `path`, or at `path`.gz when they are large; never both, so the scorer reads this run's."""
    plain, packed = path, path + ".gz"
    if len(text.encode()) > GZIP_OVER:
        part = f"{packed}.tmp{os.getpid()}"
        with open(part, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as z:
            z.write(text.encode())
        os.replace(part, packed)
        stale = plain
    else:
        S.write(plain, text)
        stale = packed
    if os.path.exists(stale):
        os.remove(stale)


def main():
    which = sys.argv[1:] or [f"Q{i}" for i in range(1, 31)]
    path = os.path.join(HERE, "results", TAG + "_agentbench_results.json")
    saved = json.load(open(path)) if os.path.exists(path) else {}
    why = S.incompatible(saved, S.configuration())  # model and request limits first: they need no Sema to compare
    config = S.configuration(S.check_build(HERE)) if not why else None
    why = why or S.incompatible(saved, config)
    if why:
        raise SystemExit(f"results/{TAG} holds a run of another configuration ({why}): nothing written; use another "
                         f"SEMA_TAG, or move results/{TAG} and {os.path.basename(path)} away first")
    missing = [q for q in which if not os.path.exists(os.path.join(HERE, "sema_queries", q + ".sql"))]
    if missing:
        raise SystemExit(f"no such query: {', '.join(missing)} (sema_queries/Q1.sql .. Q30.sql)")
    ensure_tpch()
    replays_latency = S.proxy_replays_latency()
    os.makedirs(os.path.join(HERE, "results", TAG), exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"SEMA agent_bench {S.MODEL} via meter -> {S.PROXY}  concurrency={S.CONC}  queries={which}", flush=True)
    with S.Meter(S.PROXY) as meter:
        for name in which:
            query = os.path.join(HERE, "sema_queries", name + ".sql")
            meter.reset()
            m0 = meter.upstream_misses()
            p, dt = S.run_query(S.session(meter.url) + f".read sema_queries/{name}.sql\n", "-csv", cwd=HERE)
            use = meter.snapshot()
            m1 = meter.upstream_misses()
            err = S.query_error(p, use)
            out = p.stdout.strip() if p is not None else ""
            rows = max(0, sum(1 for _ in csv.reader(io.StringIO(out))) - 1) if out else 0
            fresh = (m1 - m0) if (m0 is not None and m1 is not None) else None
            rec = {"latency_s": round(dt, 1), "llm_calls": use["requests"], "errors": use["errors"],
                   "unanswered": use["unanswered"], "llm_calls_fresh": fresh,
                   "timing": S.timing(fresh, replays_latency),
                   "tokens": use["prompt_tokens"] + use["completion_tokens"],
                   "reasoning_tokens": use["reasoning_tokens"], "cost_usd": round(use["cost_usd"], 5),
                   "rows": rows, "query_sha256": S.sha256_of(query), "error": err, "ran_at": stamp}
            write_rows(os.path.join(HERE, "results", TAG, name + ".csv"), out + "\n")
            print(f"{name:>4} lat={dt:7.1f}s calls={rec['llm_calls']:>6} errors={rec['errors']:>4} "
                  f"fresh={str(rec['llm_calls_fresh']):>6} tok={rec['tokens']:>8} ${rec['cost_usd']:.4f} rows={rows}"
                  + (f"  ERR {err}" if err else ""), flush=True)
            old = json.load(open(path)) if os.path.exists(path) else {}
            old = {**config, "updated": stamp, "per_query": {**old.get("per_query", {}), name: rec}}
            tot = old["per_query"]
            old["total"] = {"n_queries": len(tot),
                            "latency_s": round(sum(q["latency_s"] for q in tot.values()), 1),
                            "llm_calls": sum(q["llm_calls"] for q in tot.values()),
                            "cost_usd": round(sum(q["cost_usd"] for q in tot.values()), 4),
                            "tokens": sum(q["tokens"] for q in tot.values()),
                            "errors": sum(1 for q in tot.values() if q["error"]),
                            "failed_requests": sum(q["errors"] for q in tot.values()),
                            "unanswered_requests": sum(q["unanswered"] for q in tot.values())}
            S.write(path, json.dumps(old, indent=1))
    print("\nTOTAL", json.load(open(path))["total"])


if __name__ == "__main__":
    main()
