#!/usr/bin/env python3
"""Benchmark ai_filter reordering (DUCKDB_AI_REORDER) vs the query's natural order on gpt-5-mini.

Each selected govreport query runs twice against the same DuckDB build:
  OFF  -> filters in the query's written order (DuckDB's normal conjunction execution)
  ON   -> DUCKDB_AI_REORDER=1: one per-row-adaptive ai_predicate that orders each row's leaves
Both are correctness-preserving, so the result id-set must match. We report, OFF vs ON:
  chat_calls        -- gpt-5-mini calls (from the cache proxy; embeddings bypass it, so chat-only)
  prompt_tokens     -- chat input tokens  (proxy, chat-only)
  completion_tokens -- chat output+reasoning tokens (proxy, chat-only)
  cost_usd          -- $ at OpenAI (ai_usage; CPU embeddings are $0, so this is chat-only)
  latency_s         -- wall-clock of the filtering statement (embed CPU time included for ON)
Chat goes through the persistent global cache proxy, so each unique prompt is paid once and re-runs
are $0; the cache replays the original latency on hits, so ON (which reuses OFF's prompts) is not
artificially fast. DuckDB's in-process per-query cache has no latency; every doc's summary is unique.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))  # repo root (bench/ is one level down)
BIN = os.environ.get("DUCKDB_BIN", os.path.join(ROOT, "build/release/duckdb"))

ENV_COMMON = {
    "AI_PROXY_URL": os.environ.get("BENCH_PROXY", "http://localhost:4001"),  # global cache -> litellm
    "AI_MODEL": "gpt-5-mini",
    "AI_EMBED_URL": "http://localhost:4002",  # local CPU embeddings, $0, bypass the chat proxy
    "AI_EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "AI_MAX_CONCURRENCY": os.environ.get("BENCH_CONCURRENCY", "20"),
    "AI_MAX_RETRIES": "8",
    "AI_TIMEOUT_MS": "120000",
}

TABLE = os.environ.get("BENCH_TABLE", "govreport")
LIMIT = os.environ.get("BENCH_LIMIT")
LOAD = f"CREATE TABLE {TABLE} AS SELECT id, summary, source FROM read_csv('{HERE}/govreport_summary.csv', header=true)"
if LIMIT:
    LOAD += f" LIMIT {int(LIMIT)}"
LOAD += ";"



def _duck(script, env_extra=None):
    env = dict(os.environ); env.update(ENV_COMMON)
    if env_extra:
        env.update(env_extra)
    # ai_reorder has defaulted ON since 2026-08-06 (=off opts out), so OFF must be set explicitly:
    # unsetting the variable would silently run the reordered plan under the "OFF" label.
    if not env_extra or "DUCKDB_AI_REORDER" not in env_extra:
        env["DUCKDB_AI_REORDER"] = "off"
    p = subprocess.run([BIN, "-json", "-c", script], env=env, capture_output=True, text=True, timeout=14400)
    if p.returncode != 0:
        raise RuntimeError(f"duckdb failed: {p.stderr[:800]}")
    dec, objs, s, i = json.JSONDecoder(), [], p.stdout.strip(), 0
    while i < len(s):
        while i < len(s) and s[i] in " \n\r\t":
            i += 1
        if i >= len(s):
            break
        val, i = dec.raw_decode(s, i)
        objs.append(val)
    return [r for arr in objs for r in (arr if isinstance(arr, list) else [arr])]


def measure_overhead():
    # Wall-clock of LOAD + a no-AI query, to subtract the fixed CSV-read/startup cost from latency.
    t0 = time.perf_counter()
    _duck(f"{LOAD} SELECT count(*) AS n FROM {TABLE};")
    return time.perf_counter() - t0


def run(where, reorder, overhead):
    # Metrics come from ai_usage() INSIDE the duckdb process (process-isolated), so a stray/orphaned
    # process hitting the shared cache proxy can never contaminate the counts. Chat calls = total
    # llm_calls minus embed_calls (embeddings are internal to ai_filter_with_embed); prompt tokens
    # likewise subtract embed tokens. cost_usd is chat-only (CPU embeddings are $0).
    q = f"SELECT id FROM {TABLE} WHERE {where}"
    script = (
        f"{LOAD}\n"
        f"CREATE TEMP TABLE _res AS {q};\n"
        "SELECT count(*) AS n, coalesce(md5(string_agg(id, ',' ORDER BY id)), '') AS checksum FROM _res;\n"
        "SELECT coalesce(sum(llm_calls),0) AS llm_calls, coalesce(sum(embed_calls),0) AS embed_calls, "
        "coalesce(sum(input_tokens),0) AS input_tokens, coalesce(sum(embed_tokens),0) AS embed_tokens, "
        "coalesce(sum(output_tokens),0) AS output_tokens, coalesce(sum(cost_usd),0.0) AS cost_usd FROM ai_usage();\n"
    )
    t0 = time.perf_counter()
    rows = _duck(script, {"DUCKDB_AI_REORDER": "1"} if reorder else None)
    wall = time.perf_counter() - t0
    res = next(r for r in rows if "n" in r)
    u = next(r for r in rows if "llm_calls" in r)
    return {
        "n": int(res["n"]), "checksum": res["checksum"],
        "chat_calls": int(u["llm_calls"]) - int(u["embed_calls"]),
        "prompt_tokens": int(u["input_tokens"]) - int(u["embed_tokens"]),
        "completion_tokens": int(u["output_tokens"]),
        "cost_usd": float(u["cost_usd"]),
        "latency_s": max(0.0, wall - overhead),
    }


def main():
    manifest = json.load(open(f"{HERE}/manifest.json"))
    selected = json.load(open(f"{HERE}/selected.json"))
    # BENCH_ALL=1 runs every query in the manifest (45 per set = 135); otherwise the 9 selected.
    run_all = os.environ.get("BENCH_ALL", "") not in ("", "0", "false")
    docs = _duck(f"{LOAD} SELECT count(*) AS n FROM {TABLE};")[-1]["n"]
    overhead = measure_overhead()
    sys.stderr.write(f"docs={docs} fixed_overhead={overhead:.2f}s concurrency={ENV_COMMON['AI_MAX_CONCURRENCY']} "
                     f"queries={'ALL(135)' if run_all else 'selected(9)'}\n")

    results = []
    for setname in ["conjunction", "disjunction", "mix"]:
        by_q = {e["q"]: e for e in manifest[setname]}
        qlist = sorted(by_q) if run_all else selected[setname]
        for q in qlist:
            entry = by_q[q]; where = entry["sql"]; nl = entry["n_leaves"]
            tag = f"{setname[:4]}/q{q}({nl}L)"
            sys.stderr.write(f"[run] {tag} OFF ...\n"); sys.stderr.flush()
            off = run(where, False, overhead)
            sys.stderr.write(f"[run] {tag} ON  ...\n"); sys.stderr.flush()
            on = run(where, True, overhead)
            row = {"set": setname, "q": q, "n_leaves": nl, "off": off, "on": on,
                   "match": off["checksum"] == on["checksum"]}
            results.append(row)
            sys.stderr.write(
                f"    match={row['match']} rows={off['n']}/{on['n']} "
                f"calls {off['chat_calls']}->{on['chat_calls']}  "
                f"cost ${off['cost_usd']:.4f}->${on['cost_usd']:.4f}  "
                f"lat {off['latency_s']:.1f}->{on['latency_s']:.1f}s\n"); sys.stderr.flush()

    json.dump({"docs": docs, "concurrency": ENV_COMMON["AI_MAX_CONCURRENCY"], "results": results},
              open(f"{HERE}/bench_results.json", "w"), indent=1)
    report(docs, results)


def report(docs, results):
    def pct(o, n):
        return 100.0 * (o - n) / o if o else 0.0
    print(f"\n=== govreport reorder benchmark: reorder OFF (written order) vs ON (per-row DP), "
          f"docs={docs}, gpt-5-mini ===")
    cols = ["query", "match", "rows", "calls_off", "calls_on", "d%", "ptok_off", "ptok_on",
            "ctok_off", "ctok_on", "cost_off", "cost_on", "lat_off", "lat_on"]
    w = [13, 5, 5, 9, 8, 5, 8, 8, 8, 8, 8, 8, 7, 7]
    line = "  ".join(c.ljust(x) for c, x in zip(cols, w))
    print(line); print("-" * len(line))
    tot = {k: [0, 0] for k in ("calls", "pt", "ct", "cost", "lat")}
    for r in results:
        o, n = r["off"], r["on"]
        for k, a, b in (("calls", o["chat_calls"], n["chat_calls"]), ("pt", o["prompt_tokens"], n["prompt_tokens"]),
                        ("ct", o["completion_tokens"], n["completion_tokens"]), ("cost", o["cost_usd"], n["cost_usd"]),
                        ("lat", o["latency_s"], n["latency_s"])):
            tot[k][0] += a; tot[k][1] += b
        vals = [f"{r['set'][:4]}/q{r['q']}({r['n_leaves']}L)", str(r["match"]), str(o["n"]),
                str(o["chat_calls"]), str(n["chat_calls"]), f"{pct(o['chat_calls'], n['chat_calls']):.0f}",
                str(o["prompt_tokens"]), str(n["prompt_tokens"]), str(o["completion_tokens"]), str(n["completion_tokens"]),
                f"${o['cost_usd']:.4f}", f"${n['cost_usd']:.4f}", f"{o['latency_s']:.1f}", f"{n['latency_s']:.1f}"]
        print("  ".join(v.ljust(x) for v, x in zip(vals, w)))
    print("-" * len(line))
    tv = ["TOTAL", "", "", str(tot["calls"][0]), str(tot["calls"][1]), f"{pct(*tot['calls']):.0f}",
          str(tot["pt"][0]), str(tot["pt"][1]), str(tot["ct"][0]), str(tot["ct"][1]),
          f"${tot['cost'][0]:.4f}", f"${tot['cost'][1]:.4f}", f"{tot['lat'][0]:.1f}", f"{tot['lat'][1]:.1f}"]
    print("  ".join(v.ljust(x) for v, x in zip(tv, w)))
    print(f"\nSummary: chat calls {tot['calls'][0]} -> {tot['calls'][1]} ({pct(*tot['calls']):+.1f}%),  "
          f"cost ${tot['cost'][0]:.4f} -> ${tot['cost'][1]:.4f} ({pct(*tot['cost']):+.1f}%),  "
          f"completion tokens {tot['ct'][0]} -> {tot['ct'][1]} ({pct(*tot['ct']):+.1f}%),  "
          f"latency {tot['lat'][0]:.0f}s -> {tot['lat'][1]:.0f}s ({pct(*tot['lat']):+.1f}%)")
    print(f"wrote {HERE}/bench_results.json")


if __name__ == "__main__":
    main()
