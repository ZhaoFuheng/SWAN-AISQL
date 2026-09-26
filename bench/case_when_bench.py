#!/usr/bin/env python3
"""CASE-WHEN-with-AI benchmark: reorder OFF vs ON (DUCKDB_AI_REORDER), same build.

Motivation: an AI boolean tree inside a CASE `WHEN` condition (e.g. `WHEN ai_filter(h) AND ai_score(s)>0.5`)
is evaluated by DuckDB's native conjunction `Select` -- which already short-circuits + reorders the AND
children by observed selectivity (chunk-level, cost-blind, no cross-row dedup). This bench measures whether
folding that WHEN tree into ONE `ai_function_with_embed` node (per-row MLP ordering + cost-aware DP +
cache-fold + single-flight dedup) reduces chat calls / latency over the native path.

  off = DUCKDB_AI_REORDER unset  -> native CASE (adaptive-filter short-circuit)
  on  = DUCKDB_AI_REORDER=1      -> AI trees in WHEN conditions folded into ai_function_with_embed

5 representative CASE shapes (2-3 AI leaves per WHEN). Each query returns (k,v); the OFF vs ON (k,v) set
MUST match (reordering is answer-preserving) -- the correctness guard. Metrics from ai_usage() inside the
process: chat_calls = llm_calls - embed_calls (embeddings are internal to the node). Chat replays the
persistent proxy cache, so re-runs are ~$0 and ON reuses OFF's prompts (not artificially fast).
Env: CASE_DOCS (default 100)  BENCH_PROXY  BENCH_CONCURRENCY.
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
    "AI_PROXY_URL": os.environ.get("BENCH_PROXY", "http://localhost:4001"),
    "AI_MODEL": "gpt-5-mini",
    "AI_EMBED_URL": "http://localhost:4002",
    "AI_EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "AI_MAX_CONCURRENCY": os.environ.get("BENCH_CONCURRENCY", "20"),
    "AI_MAX_RETRIES": "8",
    "AI_TIMEOUT_MS": "120000",
}
DOCS = int(os.environ.get("CASE_DOCS", "100"))
LOAD = (f"CREATE TABLE gov AS SELECT row_number() OVER () AS rn, id, summary, "
        f"split_part(source,'_',1) AS org FROM read_csv('{HERE}/govreport_summary.csv', header=true) "
        f"LIMIT {DOCS};\n")

# Cached predicates (same text limit_bench / the UC benches warmed -> replay ~$0).
H = ("ai_filter('Does the document report on people''s health or medical services? Answer yes or no "
     "for the following government report: ' || summary)")
S = ("ai_filter('Does the document report on national security? Answer yes or no for the following "
     "government report: ' || summary)")
SC = "ai_score(summary, 'overall importance to the public') > 0.5"
CATS = "['health', 'security', 'economy', 'other']"


def C(cat):
    return f"ai_classify(summary, {CATS}) = '{cat}'"


def proj(case_expr):  # projection: one (k,v) row per doc
    return f"SELECT id::VARCHAR AS k, ({case_expr})::VARCHAR AS v FROM gov"


def agg(case_expr):  # conditional aggregate: one (k,v) row total
    return f"SELECT 'agg' AS k, sum({case_expr})::VARCHAR AS v FROM gov"


# (name, sql, n_ai_leaves_in_the_rewritable_WHEN)
QUERIES = [
    # 2-AI conjunction in a single WHEN
    ("and2", proj(f"CASE WHEN {H} AND {SC} THEN 'key' ELSE 'other' END"), 2),
    # deeper nested tree (AND over an OR) in the WHEN -- most room to reorder
    ("deep", proj(f"CASE WHEN {H} AND ({S} OR {SC}) THEN 'A' ELSE 'B' END"), 3),
    # chained WHENs: 1st is a 3-AI tree (rewritten), 2nd is a lone classify (native)
    ("chained", proj(f"CASE WHEN {H} AND {S} AND {SC} THEN 'all' WHEN {C('economy')} THEN 'econ' "
                     "ELSE 'other' END"), 3),
    # a cheap WHEN gates a downstream AI-tree WHEN (only rn%2=0 rows reach the AI tree)
    ("gated", proj(f"CASE WHEN rn%2=1 THEN 'skip' WHEN {H} AND {SC} THEN 'key' ELSE 'other' END"), 2),
    # conditional aggregate (sum of a CASE) -- the AI tree lives under an aggregate, not a filter/projection
    ("agg_sum", agg(f"CASE WHEN {H} AND {SC} THEN 1 ELSE 0 END"), 2),
]


def _duck(script, reorder_on):
    env = dict(os.environ)
    env.update(ENV_COMMON)
    env.pop("DUCKDB_AI_REORDER", None)
    if reorder_on:
        env["DUCKDB_AI_REORDER"] = "1"
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


def overhead():
    t0 = time.perf_counter()
    _duck(f"{LOAD} SELECT count(*) AS n FROM gov;", False)
    return time.perf_counter() - t0


def run(query, reorder_on, oh):
    script = (f"{LOAD}CREATE TEMP TABLE r AS {query};\n"
              "SELECT count(*) AS n, coalesce(md5(string_agg(k||'='||v, ',' ORDER BY k)),'') AS ck FROM r;\n"
              "SELECT coalesce(sum(llm_calls),0) AS llm, coalesce(sum(embed_calls),0) AS emb, "
              "coalesce(sum(cost_usd),0.0) AS cost FROM ai_usage();\n")
    t0 = time.perf_counter()
    rows = _duck(script, reorder_on)
    wall = time.perf_counter() - t0
    res = next(r for r in rows if "n" in r)
    u = next(r for r in rows if "llm" in r)
    return {"n": int(res["n"]), "ck": res["ck"], "calls": int(u["llm"]) - int(u["emb"]),
            "cost": float(u["cost"]), "lat": max(0.0, wall - oh)}


def main():
    if not os.path.exists(BIN):
        sys.exit(f"duckdb not found at {BIN}")
    sys.stderr.write(f"docs={DOCS} concurrency={ENV_COMMON['AI_MAX_CONCURRENCY']}\n")
    oh = overhead()
    results, mismatches = [], []
    for name, q, leaves in QUERIES:
        off = run(q, False, oh)
        on = run(q, True, oh)
        same = off["ck"] == on["ck"]
        results.append((name, leaves, off, on, same))
        if not same:
            mismatches.append(name)
        sys.stderr.write(f"[{name}] leaves={leaves}  calls {off['calls']}->{on['calls']}  "
                         f"lat {off['lat']:.1f}->{on['lat']:.1f}s  rows {off['n']}/{on['n']} match={same}\n")
        sys.stderr.flush()
    report(results)
    json.dump([{"name": n, "leaves": l, "off": o, "on": x, "match": m} for n, l, o, x, m in results],
              open(f"{HERE}/case_when_bench_results.json", "w"), indent=1)
    if mismatches:
        print(f"\n!!! {len(mismatches)} QUERIES CHANGED RESULTS (reorder must be answer-preserving): {mismatches}")
        sys.exit(1)


def pct(a, b):
    return 0.0 if a == 0 else 100.0 * (a - b) / a


def report(results):
    print(f"\n=== CASE-WHEN AI reorder: DUCKDB_AI_REORDER off vs on (docs={DOCS}, gpt-5-mini) ===")
    print(f"{'query':<10} {'leaves':<7} {'rows':<6} {'calls_off':<10} {'calls_on':<9} {'d%':<5} "
          f"{'lat_off':<8} {'lat_on':<8} {'match'}")
    print("-" * 78)
    to, tn = 0, 0
    for name, leaves, off, on, same in results:
        to += off["calls"]
        tn += on["calls"]
        print(f"{name:<10} {leaves:<7} {off['n']:<6} {off['calls']:<10} {on['calls']:<9} "
              f"{pct(off['calls'], on['calls']):<5.0f} {off['lat']:<8.1f} {on['lat']:<8.1f} {same}")
    print("-" * 78)
    print(f"total chat_calls: {to} -> {tn} ({pct(to, tn):+.0f}%)")
    print(f"\nwrote {HERE}/case_when_bench_results.json")


if __name__ == "__main__":
    main()
