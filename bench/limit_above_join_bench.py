#!/usr/bin/env python3
"""Benchmark LIMIT k that sits ABOVE a JOIN, with an AI function in play. Measures whether the AI calls
early-stop once k output rows qualify, instead of evaluating the AI function over the whole join input.

Compares two configs on the same build (both have the pull-up + reorder so the AI node exists where it can):
  base = DUCKDB_SEMANTIC_PULLUP + DUCKDB_AI_REORDER            (node present, NO limit early-stop)
  +lim = base + DUCKDB_AI_LIMIT                                (limit early-stop)
so the delta isolates the LIMIT-above-JOIN effect.

The ten queries span the distinct mechanisms (see kind). The first five isolate WHERE the AI filter lands
relative to the join; the last five add a RIGHT join, a mixed 2-leaf tree below a join, an ai_score join
condition, and two controls (k>=passing, top-K ORDER BY) that must early-stop nothing:
  BELOW_INNER  single-table ai_filter(a) under an INNER join -> DuckDB pushes it below the join; an
               a-passer can produce 0 join rows, so k a-passers != k output rows -> UNSOUND to push k below
               (must be left alone). Early-stop here would need pushing the limit through the join.
  BELOW_LEFT   single-table ai_filter(a) under a LEFT join, a on the preserved side -> every a-passer yields
               >=1 output row, so k a-passers >= k output rows -> SOUND to early-stop the below-join filter.
  JOIN_COND    multi-table ai_filter(a||b) in the join condition -> the pull-up lifts it into a FILTER above
               the join; the limit pass already pushes k into that (node above the join == output rows).
  PROJ_ABOVE   ai_classify(a) in a projection above the join -> DuckDB pushes LIMIT below the projection, so
               only k rows are ever classified (optimal already; a control that must not regress).
  BELOW_INNER_1TO1  single-table ai_filter(a) under an INNER join to a smaller dim -> DuckDB builds a
               SEMI-join reducer and keeps the ai_filter ABOVE the inner join, so the limit pass pushes k
               into that above-join node (sound: node above the join counts output rows). Shows the push
               fires whenever the filter lands above the join, not only for LEFT joins.

Correctness (per config): COMPLETENESS -- returned n == min(k, #full-join-output rows that qualify); and
res is a subset of the un-limited output. A too-eager push shows up as n < min(k, full). Metrics via
ai_usage(): chat_calls = llm_calls - embed_calls. Env: LAJ_DOCS (default 100)  BENCH_PROXY  BENCH_CONCURRENCY.
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
    "AI_MODEL": os.environ.get("BENCH_MODEL", "gpt-5-mini"),
    "AI_EMBED_URL": os.environ.get("BENCH_EMBED", "http://localhost:4002"),
    "AI_EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "AI_MAX_CONCURRENCY": os.environ.get("BENCH_CONCURRENCY", "20"),
    "AI_MAX_RETRIES": "8",
    "AI_TIMEOUT_MS": "120000",
}
DOCS = int(os.environ.get("LAJ_DOCS", "100"))
# jk = rn % 20 gives a low-cardinality join key (fan-out ~DOCS/20 on a self-join); dim keeps only keys < 10
# so a LEFT join preserves every a-row while an INNER join drops the jk>=10 half (an a-passer -> 0 rows).
LOAD = (f"CREATE TABLE gov AS SELECT row_number() OVER () AS rn, id, summary, "
        f"(row_number() OVER ())::INT % 20 AS jk FROM read_csv('{HERE}/govreport_summary.csv', header=true) "
        f"LIMIT {DOCS};\n"
        f"CREATE TABLE dim AS SELECT DISTINCT jk FROM gov WHERE jk < 10;\n"
        f"CREATE TABLE dim2 AS SELECT DISTINCT jk FROM gov WHERE jk < 5;\n")

H = ("ai_filter('Does the document report on people''s health or medical services? Answer yes or no for the "
     "following government report: ' || a.summary)")  # cached single-table health filter (on a)
HAB = ("ai_filter('Are these two government reports about a related topic? ' || substr(a.summary,1,300) || "
       "' ||| ' || substr(b.summary,1,300))")  # multi-table pair filter (bounded prompt length)
C = "ai_classify(a.summary, ['health', 'security', 'economy', 'other'])"
SC_RAW = "ai_score(a.summary, 'overall importance to the public')"  # cached 0-1 score (on a)
SC = f"{SC_RAW} > 0.5"
SCAB = ("ai_score(substr(a.summary,1,300) || ' ||| ' || substr(b.summary,1,300), "
        "'how related are these two government reports') > 0.5")  # multi-table pair score

# (name, kind, select_body_without_limit, k). k output rows requested.
QUERIES = [
    ("below_inner", "BELOW_INNER",
     f"SELECT a.id AS aid, b.id AS bid FROM gov a JOIN gov b ON a.jk = b.jk WHERE {H}", 5),
    ("below_left", "BELOW_LEFT",
     f"SELECT a.id AS aid, d.jk AS djk FROM gov a LEFT JOIN dim d ON a.jk = d.jk WHERE {H}", 5),
    ("join_cond", "JOIN_COND",
     f"SELECT a.id AS aid, b.id AS bid FROM gov a JOIN gov b ON a.jk = b.jk AND {HAB} WHERE a.id < b.id", 5),
    ("proj_above", "PROJ_ABOVE",
     f"SELECT a.id AS aid, {C} AS c FROM gov a JOIN dim d ON a.jk = d.jk", 5),
    ("below_inner_1to1", "BELOW_INNER_1TO1",
     f"SELECT a.id AS aid FROM gov a JOIN dim d ON a.jk = d.jk WHERE {H}", 5),
    # RIGHT join preserving a (the right table). DuckDB may rewrite it to a LEFT join with swapped children;
    # either way the preserved-side push fires. Exercises the RIGHT branch / the LEFT rewrite.
    ("below_right", "BELOW_RIGHT",
     f"SELECT a.id AS aid FROM dim d RIGHT JOIN gov a ON a.jk = d.jk WHERE {H}", 5),
    # A MIXED 2-leaf AI tree (ai_filter AND ai_score) below a LEFT join -> the reorder builds one node from
    # both leaves and the preserved-side push sets its limit (early-stop counts rows passing the whole tree).
    ("mixed_below_left", "BELOW_LEFT_MIXED",
     f"SELECT a.id AS aid, d.jk AS djk FROM gov a LEFT JOIN dim d ON a.jk = d.jk WHERE {H} AND {SC}", 5),
    # Multi-table ai_score (not ai_filter) in the join condition -> pull-up lifts it above the join; confirms
    # the generalized functions early-stop in the join setting too.
    ("score_join_cond", "JOIN_COND_SCORE",
     f"SELECT a.id AS aid, b.id AS bid FROM gov a JOIN gov b ON a.jk = b.jk AND {SCAB} WHERE a.id < b.id", 5),
    # CHAIN (3-way 1:1:1 INNER): gov reduced to jk<5 through TWO stacked Yannakakis SEMI reducers (gov<-dim,
    # gov<-dim2), so the ai_filter runs on the reduced set behind stacked hash joins -- the exact shape where the
    # default SEMI-reducer buffering fires. Guards that LIMIT completeness survives the buffered concurrent batch.
    ("chain_below_inner", "BELOW_INNER_CHAIN",
     f"SELECT a.id AS aid FROM gov a JOIN dim d ON a.jk = d.jk JOIN dim2 e ON a.jk = e.jk WHERE {H}", 5),
    # CONTROL k >= #passing: the push applies (LEFT join) but the limit is never hit, so ~everything is still
    # evaluated -- must show ~no reduction AND still return all passing rows (guards a bogus "always helps").
    ("k_ge_passing", "CONTROL_KGE",
     f"SELECT a.id AS aid, d.jk AS djk FROM gov a LEFT JOIN dim d ON a.jk = d.jk WHERE {H}", 1000),
    # CONTROL top-K: ORDER BY ai_score .. LIMIT over a join needs EVERY row scored to rank, so the limit must
    # NOT be pushed (the LIMIT sits over an ORDER BY / TOP_N, not a filter/join) -- expect base == +lim.
    ("orderby_topk", "CONTROL_TOPK",
     f"SELECT a.id AS aid FROM gov a LEFT JOIN dim d ON a.jk = d.jk ORDER BY {SC_RAW} DESC", 5),
]

BASE = {"DUCKDB_SEMANTIC_PULLUP": "1", "DUCKDB_AI_REORDER": "1"}
PLUS = {**BASE, "DUCKDB_AI_LIMIT": "1"}


def _duck(script, env_extra):
    env = dict(os.environ)
    env.update(ENV_COMMON)
    for k in ("DUCKDB_AI_LIMIT", "DUCKDB_AI_REORDER", "DUCKDB_SEMANTIC_PULLUP"):
        env.pop(k, None)
    env.update(env_extra)
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
    _duck(f"{LOAD} SELECT count(*) AS n FROM gov;", BASE)
    return time.perf_counter() - t0


def full_count(body, k):
    # #rows the un-limited query returns == the ground-truth passing count (evaluated once, then cached).
    rows = _duck(f"{LOAD}CREATE TEMP TABLE fullr AS {body};\nSELECT count(*) AS n FROM fullr;\n", BASE)
    return int(next(r for r in rows if "n" in r)["n"])


def run(body, k, env_extra, full_n, oh):
    stmts = [
        f"{LOAD}CREATE TEMP TABLE res AS {body} LIMIT {k};",
        "SELECT count(*) AS n FROM res;",
        "SELECT coalesce(sum(llm_calls),0) AS llm, coalesce(sum(embed_calls),0) AS emb, "
        "coalesce(sum(cost_usd),0.0) AS cost FROM ai_usage();",
        # subset check runs AFTER the ai_usage() snapshot so its cached re-eval doesn't pollute the metrics
        f"CREATE TEMP TABLE fullr AS {body};",
        "SELECT count(*) AS unsound FROM (SELECT * FROM res EXCEPT SELECT * FROM fullr);",
    ]
    t0 = time.perf_counter()
    rows = _duck("\n".join(stmts) + "\n", env_extra)
    wall = time.perf_counter() - t0
    n = int(next(r for r in rows if "n" in r)["n"])
    u = next(r for r in rows if "llm" in r)
    unsound = int(next(r for r in rows if "unsound" in r)["unsound"])
    return {"n": n, "calls": int(u["llm"]) - int(u["emb"]), "cost": float(u["cost"]),
            "lat": max(0.0, wall - oh), "ok": unsound == 0 and n == min(k, full_n)}


def main():
    if not os.path.exists(BIN):
        sys.exit(f"duckdb not found at {BIN}")
    sys.stderr.write(f"docs={DOCS} concurrency={ENV_COMMON['AI_MAX_CONCURRENCY']}\n")
    oh = overhead()
    results, bad = [], []
    for name, kind, body, k in QUERIES:
        fn = full_count(body, k)
        base = run(body, k, BASE, fn, oh)
        plus = run(body, k, PLUS, fn, oh)
        results.append((name, kind, k, fn, base, plus))
        if not (base["ok"] and plus["ok"]):
            bad.append(name)
        sys.stderr.write(f"[{name}] {kind}  full={fn}  base: {base['calls']} calls {base['lat']:.1f}s  "
                         f"+lim: {plus['calls']} calls {plus['lat']:.1f}s  n={base['n']}/{plus['n']} "
                         f"ok={base['ok'] and plus['ok']}\n")
        sys.stderr.flush()
    report(results)
    json.dump([{"name": n, "kind": ki, "k": k, "full": fn, "base": b, "plus": p}
               for n, ki, k, fn, b, p in results],
              open(f"{HERE}/limit_above_join_bench_results.json", "w"), indent=1)
    if bad:
        print(f"\n!!! {len(bad)} QUERIES FAILED CORRECTNESS (n != min(k, passing)): {bad}")
        sys.exit(1)


def pct(a, b):
    return 0.0 if a == 0 else 100.0 * (b - a) / a


def report(results):
    print(f"\n=== LIMIT above JOIN: base vs +DUCKDB_AI_LIMIT (docs={DOCS}, gpt-5-mini) ===")
    print(f"{'query':<18} {'kind':<17} {'k/full':<8} {'calls_base':<11} {'calls_lim':<10} {'dcalls':<7} "
          f"{'lat_base':<9} {'lat_lim':<9} {'ok'}")
    print("-" * 100)
    for name, kind, k, fn, b, p in results:
        print(f"{name:<18} {kind:<17} {str(k)+'/'+str(fn):<8} {b['calls']:<11} {p['calls']:<10} "
              f"{pct(b['calls'], p['calls']):<7.0f} {b['lat']:<9.1f} {p['lat']:<9.1f} {b['ok'] and p['ok']}")
    print("\ndcalls = (lim - base)/base * 100 -- negative = fewer AI calls with the limit push.")
    print(f"wrote {HERE}/limit_above_join_bench_results.json")


if __name__ == "__main__":
    main()
