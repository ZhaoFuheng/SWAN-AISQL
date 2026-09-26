#!/usr/bin/env python3
"""Benchmark pushing LIMIT k into AI functions: stop evaluating the AI function once k rows qualify,
instead of evaluating the whole scan and letting LIMIT discard the rest.

Compares DUCKDB_AI_LIMIT off vs on against the same DuckDB build (result-preserving: LIMIT k returns the
same first-k rows in scan order, just with fewer AI calls). Reports chat_calls + latency per query.

  TARGET queries  -- `WHERE ai_filter(...) LIMIT k` / projection `ai_classify(...) LIMIT k`: early
                     termination CAN stop after ~k/selectivity docs. Expect fewer calls + lower latency.
  CONTROL query   -- `ORDER BY ai_score(...) LIMIT k` (top-K): needs EVERY row scored to rank them, so
                     LIMIT CANNOT be pushed into the AI function. Expect no change (guards against a
                     wrong rewrite that breaks top-K).

Correctness (per query, on the AI_LIMIT-on output -- this is the guard the early-stop must not violate):
  soundness     -- re-evaluate the WHERE predicate (with the real scalar ai_* functions) on exactly the
                   returned rows; every one must satisfy it. Catches a node whose baked prompt/meta
                   diverges from the scalar function.
  completeness  -- returned count == min(k, #passing), where #passing is the full (un-limited) predicate
                   count. Catches an early-stop that fires before k rows actually survive the whole filter.
  match         -- the returned id-set equals the AI_LIMIT-off (ground-truth) id-set.
Any failure prints `!! VERIFY FAIL` and the script exits non-zero.

Metrics via ai_usage() inside the duckdb process. Filter predicates reuse the cached govreport health/
security prompts (replay ~$0); ai_classify/ai_score prompts are new (cached after the first run). The
verification queries run AFTER the ai_usage() snapshot, so their (cached) calls do not pollute the metrics.
Env: LIMIT_DOCS (default 100)  BENCH_PROXY  BENCH_CONCURRENCY.
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
DOCS = int(os.environ.get("LIMIT_DOCS", "100"))
LOAD = (f"CREATE TABLE gov AS SELECT row_number() OVER () AS rn, id, summary, "
        f"split_part(source,'_',1) AS org FROM read_csv('{HERE}/govreport_summary.csv', header=true) "
        f"LIMIT {DOCS};\n")

# Cached filter predicates (same text the UC benches warmed, so replay is ~$0).
H = ("ai_filter('Does the document report on people''s health or medical services? Answer yes or no "
     "for the following government report: ' || summary)")
S = ("ai_filter('Does the document report on national security? Answer yes or no for the following "
     "government report: ' || summary)")
CATS = "['health', 'security', 'economy', 'other']"
SC = "ai_score(summary, 'overall importance to the public')"  # 0-1 default; prompts cached from the control
CLASSIFY = f"ai_classify(summary, {CATS}) = 'health'"
CLASSIFY_IN = f"ai_classify(summary, {CATS}) IN ('health', 'security')"
COMPLETE = "ai_complete('In one word, the primary topic of this report: ' || summary) = 'defense'"


def where_q(pred, k):
    return f"SELECT id FROM gov WHERE {pred} LIMIT {k}"


# (name, sql, kind, pred, k). kind: TARGET (LIMIT should reduce AI calls) or CONTROL (must not).
#   pred = the WHERE predicate to re-verify; None for a projection / order-by (no WHERE) -- there the only
#          correctness property is the row count (min(k, #docs)).
QUERIES = [
    ("filter_limit5", where_q(H, 5), "TARGET", H, 5),
    ("filter_limit25", where_q(H, 25), "TARGET", H, 25),
    ("two_filter_limit5", where_q(f"{H} AND {S}", 5), "TARGET", f"{H} AND {S}", 5),
    ("classify_proj_limit5",
     f"SELECT id, ai_classify(summary, {CATS}) AS c FROM gov LIMIT 5", "TARGET", None, 5),
    ("orderby_score_limit5", f"SELECT id FROM gov ORDER BY {SC} DESC LIMIT 5", "CONTROL", None, 5),
    # Generalized AI functions in a WHERE with LIMIT. The old ai_filter-only pass did NOT early-stop these
    # (off == on); once AILimitPushdown is generalized they become real TARGETs (on < off), still verified.
    ("classify_limit5", where_q(CLASSIFY, 5), "TARGET", CLASSIFY, 5),
    ("score_limit5", where_q(f"{SC} > 0.5", 5), "TARGET", f"{SC} > 0.5", 5),
    ("complete_limit5", where_q(COMPLETE, 5), "TARGET", COMPLETE, 5),
    ("mixed_F_score_limit5", where_q(f"{H} AND {SC} > 0.5", 5), "TARGET", f"{H} AND {SC} > 0.5", 5),
    ("classify_in_limit5", where_q(CLASSIFY_IN, 5), "TARGET", CLASSIFY_IN, 5),
    # k >= passing count: the push applies but must still evaluate ~everything, so expect ~no reduction
    # (guards against a bogus "always helps").
    ("score_gt0_limit200", where_q(f"{SC} > 0.0", 200), "TARGET", f"{SC} > 0.0", 200),
]


def _duck(script, env_extra):
    env = dict(os.environ)
    env.update(ENV_COMMON)
    env.pop("DUCKDB_AI_LIMIT", None)
    if env_extra:
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
    _duck(f"{LOAD} SELECT count(*) AS n FROM gov;", None)
    return time.perf_counter() - t0


def run(query, limit_on, oh, verify=None):
    stmts = [
        f"{LOAD}CREATE TEMP TABLE _res AS {query};",
        "SELECT count(*) AS n, coalesce(md5(string_agg(id, ',' ORDER BY id)),'') AS ck FROM _res;",
        "SELECT coalesce(sum(llm_calls),0) AS llm, coalesce(sum(embed_calls),0) AS emb, "
        "coalesce(sum(cost_usd),0.0) AS cost FROM ai_usage();",
    ]
    if verify is not None:
        pred, _k = verify
        # These run AFTER the ai_usage() snapshot above, so their (cached) AI calls stay out of the metrics.
        if pred:
            stmts.append(f"SELECT count(*) AS sound FROM gov WHERE id IN (SELECT id FROM _res) AND ({pred});")
            stmts.append(f"SELECT count(*) AS passing FROM gov WHERE ({pred});")
        else:  # no WHERE: every returned row trivially qualifies; correctness is just the count
            stmts.append("SELECT count(*) AS sound FROM _res;")
            stmts.append("SELECT count(*) AS passing FROM gov;")
    t0 = time.perf_counter()
    rows = _duck("\n".join(stmts) + "\n", {"DUCKDB_AI_LIMIT": "1"} if limit_on else None)
    wall = time.perf_counter() - t0
    res = next(r for r in rows if "n" in r)
    u = next(r for r in rows if "llm" in r)
    out = {"n": int(res["n"]), "ck": res["ck"], "calls": int(u["llm"]) - int(u["emb"]),
           "cost": float(u["cost"]), "lat": max(0.0, wall - oh)}
    if verify is not None:
        out["sound"] = int(next(r for r in rows if "sound" in r)["sound"])
        out["passing"] = int(next(r for r in rows if "passing" in r)["passing"])
    return out


def main():
    if not os.path.exists(BIN):
        sys.exit(f"duckdb not found at {BIN}")
    sys.stderr.write(f"docs={DOCS} concurrency={ENV_COMMON['AI_MAX_CONCURRENCY']}\n")
    oh = overhead()
    results, failures = [], []
    for name, q, kind, pred, k in QUERIES:
        off = run(q, False, oh)
        on = run(q, True, oh, verify=(pred, k))
        exp = min(k, on["passing"])           # correct row count = min(LIMIT, #passing)
        sound_ok = on["sound"] == on["n"]     # every returned row satisfies the WHERE predicate
        count_ok = on["n"] == exp             # returned exactly min(k, #passing) rows
        match_ok = off["ck"] == on["ck"]      # optimized id-set == ground-truth (AI_LIMIT off) id-set
        ok = sound_ok and count_ok and match_ok
        results.append((name, kind, off, on, exp, ok))
        if not ok:
            failures.append(name)
        sys.stderr.write(f"[{name}] {kind}  off: {off['calls']} calls {off['lat']:.1f}s   "
                         f"on: {on['calls']} calls {on['lat']:.1f}s   rows {off['n']}/{on['n']} "
                         f"verified={ok}\n")
        if not ok:
            sys.stderr.write(f"   !! VERIFY FAIL sound={sound_ok} count={count_ok}(got {on['n']} exp {exp}) "
                             f"match={match_ok}\n")
        sys.stderr.flush()
    report(results)
    json.dump([{"name": n, "kind": k, "off": o, "on": x, "exp": e, "verified": v}
               for n, k, o, x, e, v in results],
              open(f"{HERE}/limit_bench_results.json", "w"), indent=1)
    if failures:
        print(f"\n!!! {len(failures)} QUERIES FAILED VERIFICATION: {failures}")
        sys.exit(1)


def pct(a, b):
    return 0.0 if a == 0 else 100.0 * (a - b) / a


def report(results):
    print(f"\n=== LIMIT push-down into AI functions: DUCKDB_AI_LIMIT off vs on (docs={DOCS}, gpt-5-mini) ===")
    print(f"{'query':<22} {'kind':<8} {'rows':<7} {'calls_off':<10} {'calls_on':<9} {'d%':<5} "
          f"{'lat_off':<8} {'lat_on':<8} {'verified'}")
    print("-" * 92)
    for name, kind, off, on, exp, ok in results:
        print(f"{name:<22} {kind:<8} {str(off['n'])+'/'+str(on['n']):<7} {off['calls']:<10} {on['calls']:<9} "
              f"{pct(off['calls'], on['calls']):<5.0f} {off['lat']:<8.1f} {on['lat']:<8.1f} {ok}")
    print(f"\nwrote {HERE}/limit_bench_results.json")


if __name__ == "__main__":
    main()
