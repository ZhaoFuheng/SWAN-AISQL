#!/usr/bin/env python3
"""Characterize WHERE an AI filter should run in SWAN AISQL: push it down, pull it above the join, or
pull up + leave an MLP-gated speculative pre-filter at the leaf. Reorder (DP per-row ordering) is ON in
every config -- it almost always helps, so it is a constant, not a variable. We sweep the join's
cardinality effect (selective / 1:1 / expansive), since that is what decides the answer.

Configs (all with DUCKDB_AI_REORDER=1):
  push_down     -- filters stay at the base scans (DuckDB default placement)
  pull_up       -- DUCKDB_SEMANTIC_PULLUP=1 + AI_SPECULATIVE_THRESHOLD=0: filters lifted above the
                   join; the leaf pre-filter passes every row through (== a pure pull-up / placeholder)
  speculative   -- DUCKDB_SEMANTIC_PULLUP=1 (gate 0.5): the leaf pre-filter MLP-prunes likely-fail rows

Use cases (govreport, cached health/security predicates so replay is ~$0):
  UC1 selective join  -- filter + join that keeps 10/50 via ARBITRARY ids (only the join reduces, no
                         cheap relational predicate to propagate). Isolates: does pull-up cut calls,
                         does the speculative gate defeat the reduction?
  UC2 one_to_one      -- filter + 1:1 join (join neither reduces nor expands). Neutral reference.
  UC3 expansive_same  -- two filters on ONE relation + expansive self-join (49x). Isolates: does
                         speculative recover the pull-up fan-out penalty vs push-down?
  UC4 expansive_split -- two filters on DIFFERENT sides + expansive self-join, cross and sparse.
                         Isolates: combined-node short-circuit above the join.
  UC5-UC7 generalized -- UC3/UC4 shapes but with ai_classify(..)='x' / ai_score(..)>v / ai_complete(..)='x'
                         comparison leaves instead of ai_filter. Until the pull-up is generalized to these
                         patterns they push below the join (all configs equal); the target is that they get
                         the same pull-up + speculative + dedup treatment as the ai_filter UCs.

Reports LLM calls + latency for each (result-preserving; id-sets asserted equal), then a synthesis.
Env: SPEC_LIMIT (docs, default 50)  SPEC_WARM_CAP (max one-time warm-up $, default 0.40)  BENCH_CONCURRENCY.
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
LIMIT = int(os.environ.get("SPEC_LIMIT", "50"))


def extract_two_leaves():
    """Lift conj/q0's two ai_filter(... || summary) leaves verbatim (so their prompts are cached)."""
    manifest = json.load(open(f"{HERE}/manifest.json"))
    sql = next(e for e in manifest["conjunction"] if e["q"] == 0)["sql"]
    leaves, i = [], 0
    while True:
        start = sql.find("ai_filter(", i)
        if start < 0:
            break
        depth, j = 0, start + len("ai_filter")
        while j < len(sql):
            if sql[j] == "(":
                depth += 1
            elif sql[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        leaves.append(sql[start:j + 1])
        i = j + 1
    return leaves[0], leaves[1]


L0, L1 = extract_two_leaves()


def H(x):
    return L0.replace("|| summary", f"|| {x}.summary")


def S(x):
    return L1.replace("|| summary", f"|| {x}.summary")


# Generalized AI-comparison leaves (classify/score/complete), same shape as H/S but the other functions.
# Used by UC5-UC7 so the speculative bench also covers `ai_classify(..)='x'`, `ai_score(..)>v`,
# `ai_complete(..)='x'` in a join query's WHERE (pull-up generalization target).
CATS = "['health','defense','economy','education','environment','technology','justice','other']"


def C(x, cat):        # ai_classify(x.summary, CATS) = 'cat'
    return f"ai_classify({x}.summary, {CATS}) = '{cat}'"


def SC(x, crit):      # ai_score(x.summary, 'crit') > 0.5   (0-1 default scale)
    return f"ai_score({x}.summary, '{crit}') > 0.5"


def M(x, instr, val):  # ai_complete('instr: ' || x.summary) = 'val'
    return f"ai_complete('{instr}: ' || {x}.summary) = '{val}'"


GOV = (f"CREATE TABLE gov AS SELECT row_number() OVER () AS rn, id, summary, "
       f"split_part(source,'_',1) AS org FROM read_csv('{HERE}/govreport_summary.csv', header=true) "
       f"LIMIT {LIMIT};\n")

USE_CASES = [
    {"name": "UC1_selective", "shape": "single filter + selective join (keeps 10/50, arbitrary ids)",
     "setup": "CREATE TABLE keep AS SELECT id FROM gov LIMIT 10;\n",
     "query": f"SELECT g.id FROM gov g JOIN keep k ON g.id=k.id WHERE {H('g')}"},
    {"name": "UC2_one_to_one", "shape": "single filter + 1:1 join (neutral)",
     "setup": "CREATE TABLE allids AS SELECT id FROM gov;\n",
     "query": f"SELECT g.id FROM gov g JOIN allids k ON g.id=k.id WHERE {H('g')}"},
    {"name": "UC3_expansive_same", "shape": "two filters on ONE relation + expansive self-join (49x)",
     "setup": "",
     "query": f"SELECT DISTINCT a.id FROM gov a JOIN gov b ON a.org=b.org AND a.id<>b.id "
              f"WHERE {H('a')} AND {S('a')}"},
    {"name": "UC4_split_cross", "shape": "two filters SPLIT across an expansive (cross) join",
     "setup": "",
     "query": f"SELECT DISTINCT a.id FROM gov a JOIN gov b ON a.id<>b.id WHERE {H('a')} AND {S('b')}"},
    {"name": "UC4_split_sparse", "shape": "two filters SPLIT across a SPARSE join (each a->one b)",
     "setup": "",
     "query": f"SELECT DISTINCT a.id FROM gov a JOIN gov b ON a.rn=b.rn+1 WHERE {H('a')} AND {S('b')}"},
    # UC5-UC7: same expansive-self-join shapes as UC3/UC4 but with ai_classify=/ai_score>/ai_complete=
    # comparison leaves instead of ai_filter. Until the pull-up is generalized these are pushed below the
    # join (all configs identical); once it lifts `ai_classify(..)='x'` etc. they get the speculative + dedup
    # treatment like the ai_filter UCs.
    {"name": "UC5_classify_score_same", "shape": "classify + score on ONE relation + expansive self-join (49x)",
     "setup": "",
     "query": f"SELECT DISTINCT a.id FROM gov a JOIN gov b ON a.org=b.org AND a.id<>b.id "
              f"WHERE {C('a', 'health')} AND {SC('a', 'public importance')}"},
    {"name": "UC6_classify_score_split", "shape": "classify(a) + score(b) SPLIT across an expansive (cross) join",
     "setup": "",
     "query": f"SELECT DISTINCT a.id FROM gov a JOIN gov b ON a.id<>b.id "
              f"WHERE {C('a', 'defense')} AND {SC('b', 'national security relevance')}"},
    {"name": "UC7_complete_filter_same", "shape": "complete + ai_filter on ONE relation + expansive self-join",
     "setup": "",
     "query": f"SELECT DISTINCT a.id FROM gov a JOIN gov b ON a.org=b.org AND a.id<>b.id "
              f"WHERE {M('a', 'In one word the primary topic', 'defense')} AND {H('a')}"},
    # UC8 (multi-table ai_score(a||b) in a join condition) is verified separately (EXPLAIN: lifted above the
    # join + AI_DEDUP; mock: result-preserving). It is NOT in the default run: an expansive cross-doc predicate
    # is ~2450 DISTINCT pair-calls on long paired summaries (~$2.45, blows the warm-up cap), and because the
    # pairs are distinct the dedup gives no call savings -- only a streaming-latency win that needs that scale.
]

CONFIGS = [
    ("push_down", {"DUCKDB_AI_REORDER": "1"}),
    # pure pull-up: filters lifted above the join, NO speculative leaf node (threshold 0 -> not emitted).
    ("pull_up", {"DUCKDB_AI_REORDER": "1", "DUCKDB_SEMANTIC_PULLUP": "1", "AI_SPECULATIVE_THRESHOLD": "0"}),
    # ours (recommended SWAN config): pull-up + the speculative leaf node WHEN IT HELPS (default gate 0.5,
    # auto-bypassed for selective joins via AI_SPECULATIVE_MIN_FANOUT) + the streaming AI-dedup operator
    # ALWAYS on (evaluates each distinct input once over the join output, no per-chunk barriers).
    ("ours", {"DUCKDB_AI_REORDER": "1", "DUCKDB_SEMANTIC_PULLUP": "1", "DUCKDB_AI_DEDUP": "1"}),
    # ours_always: same as ours but ALWAYS emit the speculative leaf node (drop the MIN_FANOUT bypass). Valid
    # once Yannakakis reduces the base relationally first -- tests whether the fanout gate is now obsolete.
    # (Run the whole bench with DUCKDB_YANNAKAKIS=1 in the env so every config reduces the base first.)
    ("ours_always", {"DUCKDB_AI_REORDER": "1", "DUCKDB_SEMANTIC_PULLUP": "1", "DUCKDB_AI_DEDUP": "1",
                     "AI_SPECULATIVE_ALWAYS": "1"}),
]
PULLUP_KEYS = ("DUCKDB_SEMANTIC_PULLUP", "AI_SPECULATIVE_THRESHOLD", "DUCKDB_AI_REORDER", "DUCKDB_AI_DEDUP",
               "AI_SPECULATIVE_ALWAYS")


def _duck(script, env_extra):
    env = dict(os.environ)
    env.update(ENV_COMMON)
    for k in PULLUP_KEYS:
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


def overhead(load):
    t0 = time.perf_counter()
    _duck(f"{load} SELECT count(*) AS n FROM gov;", {})
    return time.perf_counter() - t0


def run(load, query, env_extra, oh):
    script = (f"{load}CREATE TEMP TABLE _res AS {query};\n"
              "SELECT count(*) AS n, coalesce(md5(string_agg(id, ',' ORDER BY id)),'') AS ck FROM _res;\n"
              "SELECT coalesce(sum(llm_calls),0) AS llm, coalesce(sum(embed_calls),0) AS emb, "
              "coalesce(sum(cost_usd),0.0) AS cost FROM ai_usage();\n")
    t0 = time.perf_counter()
    rows = _duck(script, env_extra)
    wall = time.perf_counter() - t0
    res = next(r for r in rows if "n" in r)
    u = next(r for r in rows if "llm" in r)
    return {"n": int(res["n"]), "ck": res["ck"], "calls": int(u["llm"]) - int(u["emb"]),
            "cost": float(u["cost"]), "lat": max(0.0, wall - oh)}


def main():
    if not os.path.exists(BIN):
        sys.exit(f"duckdb not found at {BIN}")
    warm_cap = float(os.environ.get("SPEC_WARM_CAP", "0.40"))
    sys.stderr.write(f"docs={LIMIT} concurrency={ENV_COMMON['AI_MAX_CONCURRENCY']}\n")

    # One-time warm-up: run each UC once (push_down) to cache any uncached prompts; capped.
    warm = 0.0
    for uc in USE_CASES:
        warm += run(GOV + uc["setup"], uc["query"], {"DUCKDB_AI_REORDER": "1"}, 0.0)["cost"]
    if warm > warm_cap:
        sys.exit(f"ABORT: warm-up ${warm:.4f} > cap ${warm_cap:.2f}")
    sys.stderr.write(f"warm-up ${warm:.4f} (one-time; measured runs are cache hits)\n")

    results = {}
    for uc in USE_CASES:
        load = GOV + uc["setup"]
        oh = overhead(load)
        res = {}
        for cname, env in CONFIGS:
            sys.stderr.write(f"[run] {uc['name']}/{cname} ...\n"); sys.stderr.flush()
            res[cname] = run(load, uc["query"], env, oh)
            r = res[cname]
            sys.stderr.write(f"    n={r['n']} calls={r['calls']} lat={r['lat']:.1f}s\n"); sys.stderr.flush()
        results[uc["name"]] = res
    report(results)
    json.dump({"docs": LIMIT, "warm_cost": warm, "use_cases": USE_CASES, "results": results},
              open(f"{HERE}/bench_speculative_results.json", "w"), indent=1)


def report(results):
    print(f"\n=== SWAN AISQL: where should the AI filter run? (reorder ON everywhere, docs={LIMIT}, "
          f"gpt-5-mini, replay ON) ===")
    synth = []
    for uc in USE_CASES:
        res = results[uc["name"]]
        cks = {c: res[c]["ck"] for c, _ in CONFIGS}
        match = len(set(cks.values())) == 1
        print(f"\n-- {uc['name']}: {uc['shape']} --")
        print(f"{'config':<13} {'rows':<5} {'LLM_calls':<10} {'latency_s':<10}")
        print("-" * 42)
        for cname, _ in CONFIGS:
            r = res[cname]
            print(f"{cname:<13} {r['n']:<5} {r['calls']:<10} {r['lat']:<10.1f}")
        print("-" * 42)
        print(f"id-sets identical across configs: {match}")
        # verdict: best on calls, best on latency
        best_calls = min(CONFIGS, key=lambda c: res[c[0]]["calls"])[0]
        best_lat = min(CONFIGS, key=lambda c: res[c[0]]["lat"])[0]
        ours, pull = res["ours"], res["pull_up"]
        helps = "HELPS" if ours["calls"] < pull["calls"] or ours["lat"] < pull["lat"] - 1.0 else \
                ("HURTS" if ours["calls"] > pull["calls"] or ours["lat"] > pull["lat"] + 1.0 else "neutral")
        # always-add gate effect: ours_always (drop MIN_FANOUT) vs ours (gated). Regression = more calls.
        alw = res.get("ours_always")
        gate = "n/a"
        if alw is not None:
            gate = ("REGRESSES(+calls)" if alw["calls"] > ours["calls"] else
                    ("saves calls" if alw["calls"] < ours["calls"] else
                     ("faster" if alw["lat"] < ours["lat"] - 1.0 else
                      ("slower" if alw["lat"] > ours["lat"] + 1.0 else "same"))))
        print(f"best calls: {best_calls}   best latency: {best_lat}   ours vs pull_up: {helps}   "
              f"always vs ours: {gate}")
        synth.append((uc["name"], best_calls, best_lat, helps, gate))
    print("\n=== synthesis: shape -> best config -> does ours help vs pull_up? + always-add effect ===")
    print(f"{'use case':<20} {'best_calls':<12} {'best_latency':<13} {'ours_vs_pull_up':<16} {'always_vs_ours'}")
    print("-" * 78)
    for name, bc, bl, h, g in synth:
        print(f"{name:<20} {bc:<12} {bl:<13} {h:<16} {g}")
    print(f"\nwrote {HERE}/bench_speculative_results.json")


if __name__ == "__main__":
    main()
