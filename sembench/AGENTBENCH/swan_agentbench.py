#!/usr/bin/env python3
"""SWAN on the PLOP agent_bench 30 queries (translations in swan_queries/QN.sql).

Per query: a fresh CLI process (query-scoped local cache by construction), routed via the
:4001 recording proxy; calls = chat only (embeds excluded, per the reporting rule); the final
SELECT's rows go to results/swan/QN.csv for cross-system agreement.

  run:  python3 swan_agentbench.py [Q1 Q2 ...]
  env:  DUCKDB_BIN, AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna), AI_MAX_CONCURRENCY (20)
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
# Provider keys come from the repo's .env (git-ignored; see .env.example): loaded here so the engine and
# the proxy see OPENAI_API_KEY / TYPESAFE_API_KEY without exporting them by hand. Existing env wins.
_ENV_FILE = os.path.join(HERE, "../../.env")
if os.path.exists(_ENV_FILE):
    for _line in open(_ENV_FILE):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
_BUILDS = [os.path.abspath(os.path.join(HERE, "../../build", b, "duckdb")) for b in ("release", "reldebug")]
BIN = os.environ.get("DUCKDB_BIN", next((p for p in _BUILDS if os.path.exists(p)), _BUILDS[0]))
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = os.environ.get("AI_MAX_CONCURRENCY", "20")
MARK = "===USAGE==="
# extra SET statements for config A/Bs (e.g. SWAN_EXTRA_SET="SET ai_inline_ai_ctes=true;")
EXTRA_SET = os.environ.get("SWAN_EXTRA_SET", "")
# results tag so an A/B run does not clobber the certified results
TAG = os.environ.get("SWAN_TAG", "swan")
# the SWAN translations of the 30 PLOP queries (SWAN_QUERY_DIR overrides for an A/B against a variant set)
QUERY_DIR = os.environ.get("SWAN_QUERY_DIR", "swan_queries")

ENV = {**os.environ, "AI_MODEL": MODEL, "AI_API_KEY": "sk-test", "AI_MAX_CONCURRENCY": CONC,
       "AI_TIMEOUT_MS": "180000"}


def csv_row_count(text):
    """Data rows in a CSV string, counted by the csv module so quoted newlines do not inflate it."""
    if not text.strip():
        return 0
    import csv as _csv
    import io as _io
    return max(0, sum(1 for _ in _csv.reader(_io.StringIO(text))) - 1)


def main():
    which = sys.argv[1:] or [f"Q{i}" for i in range(1, 31)]
    os.makedirs(os.path.join(HERE, "results", TAG), exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    per_query = {}
    for name in which:
        sql = open(os.path.join(HERE, QUERY_DIR, name + ".sql")).read()
        script = (f"SET ai_endpoint='{PROXY}';\n{EXTRA_SET}\n.mode csv\n.headers on\n" + sql +
                  f"\n.print {MARK}\n"
                  "SELECT (sum(llm_calls)-sum(embed_calls))::BIGINT AS chat, sum(cache_hits)::BIGINT AS hits,"
                  " sum(total_tokens)::BIGINT AS tok, round(sum(cost_usd),5) AS cost,"
                  " sum(hedged_calls)::BIGINT AS hedged FROM ai_usage();\n")
        t0 = time.time()
        p = subprocess.run([BIN], input=script, env=ENV, text=True, capture_output=True, cwd=HERE)
        dt = time.time() - t0
        err = None
        rows = 0
        chat = hits = tok = hedged = 0
        cost = 0.0
        if p.returncode != 0 or "Error" in p.stderr:
            err = (p.stderr.strip().split("\n")[-1] or "nonzero exit")[:300]
        out = p.stdout
        if MARK in out:
            result_part, usage_part = out.split(MARK, 1)
            usage_lines = [l for l in usage_part.strip().split("\n") if l and not l.startswith("chat")]
            if usage_lines:
                f = usage_lines[-1].split(",")
                chat, hits, tok, cost, hedged = int(f[0]), int(f[1]), int(f[2]), float(f[3]), int(f[4])
            # The result is EVERYTHING before the marker. It must not be split on blank lines:
            # a quoted field can contain them (review text routinely does), and splitting shredded
            # the CSV into fragments -- the saved file then started mid-row, failed to parse, and
            # the row count was the fragment's line count. Nothing else in the script prints, since
            # the setup statements are DDL.
            result_csv = result_part.strip("\n")
            rows = max(0, csv_row_count(result_csv))
            open(os.path.join(HERE, "results", TAG, name + ".csv"), "w").write(result_csv + "\n")
        elif not err:
            err = "no usage marker in output"
        rec = {"latency_s": round(dt, 1), "llm_calls": chat, "local_hits": hits, "tokens": tok,
               "cost_usd": cost, "hedged": hedged, "rows": rows, "error": err, "ran_at": stamp}
        per_query[name] = rec
        print(f"{name:>4} lat={dt:7.1f}s calls={chat:>6} hits={hits:>6} tok={tok:>8} ${cost:.4f} "
              f"rows={rows}" + (f"  ERR {err}" if err else ""), flush=True)
        path = os.path.join(HERE, "results", TAG + "_agentbench_results.json")
        old = json.load(open(path)) if os.path.exists(path) else {"per_query": {}}
        old["model"], old["concurrency"], old["updated"] = MODEL, CONC, stamp
        old["per_query"].update(per_query)
        tot = old["per_query"]
        old["total"] = {"n_queries": len(tot),
                        "latency_s": round(sum(q["latency_s"] for q in tot.values()), 1),
                        "llm_calls": sum(q["llm_calls"] for q in tot.values()),
                        "cost_usd": round(sum(q["cost_usd"] for q in tot.values()), 4),
                        "tokens": sum(q["tokens"] for q in tot.values()),
                        "errors": sum(1 for q in tot.values() if q["error"])}
        json.dump(old, open(path, "w"), indent=1)
    print("\nTOTAL", json.load(open(os.path.join(HERE, "results", TAG + "_agentbench_results.json")))["total"])


main()
