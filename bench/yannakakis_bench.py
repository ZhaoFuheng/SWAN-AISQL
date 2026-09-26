#!/usr/bin/env python3
"""Yannakakis benchmark: multi-way joins + selective RELATIONAL predicates + an ai_filter on a table the
joins reduce. Yannakakis (DUCKDB_YANNAKAKIS) should reduce the ai_filter's table RELATIONALLY first (no LLM
calls in the reduction), so the ai_filter runs only on the surviving rows.

Base table `docs` = 50 govreport summaries (BENCH_LIMIT overrides) + integer join keys. Two dimension styles:
  * SMALL dims (docs is the hash-join PROBE side) -> DuckDB's runtime dynamic filter already reduces docs.
  * BIG dims  (docs is the hash-join BUILD side)  -> the dynamic filter can't reduce docs, so the ai_filter
    runs on ALL docs unless Yannakakis reduces it first. This is where Yannakakis should win.

Metric: chat_calls (ai_usage llm_calls = docs the ai_filter evaluated), OFF vs ON, with the result id-set
checksum (must match -- reduction is answer-preserving) + latency.

  run (proxies up):  python3 yannakakis_bench.py            # 50 docs
                     BENCH_LIMIT=80 python3 yannakakis_bench.py
"""
import json, os, sys, time
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("b", f"{HERE}/bench.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

N = int(os.environ.get("BENCH_LIMIT", "50"))
PRED = "ai_filter('Does this report discuss the federal budget or government spending? ' || d.summary)"
CATS = "['health','defense','economy','education','environment','technology','justice','other']"
# The SAME relational reduction should shrink the AI table before ANY AI function fires -- not just ai_filter.
# These mirror star2_build (docs=BUILD, big dims g1/g2 reduce docs 50->few) but swap the AI predicate for
# ai_classify / ai_score / ai_complete, in the WHERE clause, a JOIN condition, and a projection.
CLS = f"ai_classify(d.summary, {CATS}) = 'health'"
SCO = "ai_score(d.summary, 'importance to the public') > 0.5"
CMP = "ai_complete('Does this report discuss health or medical services? Answer yes or no: ' || d.summary) = 'yes'"

# ---- schema: docs (50) with join keys + small and big dimensions + a 3-hop chain ----
SCHEMA = f"""
CREATE TABLE docs AS SELECT id, summary, (rn%20) AS k1, (rn%17) AS k2, (rn%13) AS k3 FROM
  (SELECT id, summary, row_number() OVER () AS rn FROM read_csv('{HERE}/govreport_summary.csv', header=true)) t
  WHERE rn <= {N};
-- SMALL dims: fewer rows than docs -> docs is the PROBE side (dynamic filter reduces docs)
CREATE TABLE s1 AS SELECT i AS k1, ((i%20)<4) AS hot FROM range(20) t(i);
CREATE TABLE s2 AS SELECT i AS k2, ((i%17)<4) AS hot FROM range(17) t(i);
CREATE TABLE s3 AS SELECT i AS k3, ((i%13)<3) AS hot FROM range(13) t(i);
-- BIG dims: many rows per key -> docs is the BUILD side (dynamic filter cannot reduce docs)
CREATE TABLE g1 AS SELECT (i%20) AS k1, ((i%20)<4) AS hot FROM range(2000) t(i);
CREATE TABLE g2 AS SELECT (i%17) AS k2, ((i%17)<4) AS hot FROM range(2000) t(i);
CREATE TABLE g3 AS SELECT (i%13) AS k3, ((i%13)<3) AS hot FROM range(2000) t(i);
-- CHAIN: docs -> m (mid) -> f (far); the selective predicate is at the FAR end.
CREATE TABLE m  AS SELECT i AS k1, (i%5) AS fk FROM range(20) t(i);
CREATE TABLE f  AS SELECT i AS fk, (i=0) AS hot FROM range(5) t(i);
CREATE TABLE gm AS SELECT (i%20) AS k1, (i%5) AS fk FROM range(2000) t(i);
CREATE TABLE gf AS SELECT (i%5) AS fk, ((i%5)=0) AS hot FROM range(2000) t(i);
-- DEEPER chain: docs -> ca -> cb -> cf (4 tables); predicate at cf must propagate 3 hops to docs.
CREATE TABLE ca AS SELECT (i%20) AS k1, (i%10) AS a2 FROM range(2000) t(i);
CREATE TABLE cb AS SELECT (i%10) AS a2, (i%5) AS b2 FROM range(2000) t(i);
CREATE TABLE cf AS SELECT (i%5) AS b2, ((i%5)=0) AS hot FROM range(2000) t(i);
-- CONTROL: a plain dimension with NO predicate -> no relational reduction possible.
CREATE TABLE p1 AS SELECT i AS k1 FROM range(20) t(i);
"""

QUERIES = {
  # docs=PROBE (small dims): the dynamic filter already reduces docs; current Yannakakis REGRESSES it.
  "star2_probe": f"SELECT DISTINCT d.id FROM docs d JOIN s1 ON d.k1=s1.k1 JOIN s2 ON d.k2=s2.k2 "
                 f"WHERE s1.hot AND s2.hot AND {PRED}",
  # docs=BUILD (big dims): nothing reduces docs -> ai_filter on all N unless Yannakakis. The win cases.
  "star2_build": f"SELECT DISTINCT d.id FROM docs d JOIN g1 ON d.k1=g1.k1 JOIN g2 ON d.k2=g2.k2 "
                 f"WHERE g1.hot AND g2.hot AND {PRED}",
  "star3_build": f"SELECT DISTINCT d.id FROM docs d JOIN g1 ON d.k1=g1.k1 JOIN g2 ON d.k2=g2.k2 "
                 f"JOIN g3 ON d.k3=g3.k3 WHERE g1.hot AND g2.hot AND g3.hot AND {PRED}",
  # CHAIN (big): far-end predicate must propagate through the mid table to reduce docs.
  "chain_build": f"SELECT DISTINCT d.id FROM docs d JOIN gm ON d.k1=gm.k1 JOIN gf ON gm.fk=gf.fk "
                 f"WHERE gf.hot AND {PRED}",
  # DEEPER chain: predicate 3 hops away from docs -> stresses the fixed-point propagation.
  "chain3_build": f"SELECT DISTINCT d.id FROM docs d JOIN ca ON d.k1=ca.k1 JOIN cb ON ca.a2=cb.a2 "
                  f"JOIN cf ON cb.b2=cf.b2 WHERE cf.hot AND {PRED}",
  # CONTROL: no selective relational predicate -> Yannakakis has nothing to reduce; ON must ~= OFF (no
  # regression, and no spurious semi-join that changes results).
  "no_reduce_ctrl": f"SELECT DISTINCT d.id FROM docs d JOIN p1 ON d.k1=p1.k1 WHERE {PRED}",
  # --- GENERALIZED AI functions: the relational reduction must shrink docs before classify/score/complete too ---
  # classify / score / complete as the SELECTIVE predicate in a WHERE clause (docs=BUILD, reduced by g1,g2).
  "classify_where": f"SELECT DISTINCT d.id FROM docs d JOIN g1 ON d.k1=g1.k1 JOIN g2 ON d.k2=g2.k2 "
                    f"WHERE g1.hot AND g2.hot AND {CLS}",
  "score_where": f"SELECT DISTINCT d.id FROM docs d JOIN g1 ON d.k1=g1.k1 JOIN g2 ON d.k2=g2.k2 "
                 f"WHERE g1.hot AND g2.hot AND {SCO}",
  "complete_where": f"SELECT DISTINCT d.id FROM docs d JOIN g1 ON d.k1=g1.k1 JOIN g2 ON d.k2=g2.k2 "
                    f"WHERE g1.hot AND g2.hot AND {CMP}",
  # AI comparison in a JOIN condition (single-table on d -> pushed to the docs side; reduced same as WHERE).
  "classify_joincond": f"SELECT DISTINCT d.id FROM docs d JOIN g1 ON d.k1=g1.k1 "
                       f"JOIN g2 ON d.k2=g2.k2 AND {CLS} WHERE g1.hot AND g2.hot",
  # AI function in the PROJECTION (SELECT ai_classify(..)). SMALL dims s1,s2 (1:1, no fan-out) so the projected
  # classify fires once per surviving doc -- measures whether the reduced doc-set (not all 50) reaches it.
  "classify_proj": f"SELECT DISTINCT d.id, ai_classify(d.summary, {CATS}) AS c FROM docs d "
                   f"JOIN s1 ON d.k1=s1.k1 JOIN s2 ON d.k2=s2.k2 WHERE s1.hot AND s2.hot",
  # MULTI-TABLE ai_score in a JOIN condition (a self-join pair prompt -> lifted above the join, can't be
  # pushed to one side). g1.hot reduces `a` FIRST, so the pair-wise score fires on far fewer a-rows.
  "score_join_multi": f"SELECT DISTINCT a.id FROM docs a JOIN g1 ON a.k1=g1.k1 JOIN docs b ON a.k2=b.k2 AND "
                      f"ai_score(substr(a.summary,1,100) || ' ||| ' || substr(b.summary,1,100), "
                      f"'how related are these two reports') > 0.5 WHERE g1.hot AND a.id <> b.id",
}


def run(query, env):
    # Checksum the WHOLE row (row-struct cast), not just id, so projection queries that return the AI value
    # (SELECT d.id, ai_classify(..)) are verified answer-preserving on that value too, not only the id-set.
    script = (SCHEMA + f"\nCREATE TEMP TABLE _res AS {query};\n"
              "SELECT count(*) AS n, coalesce(md5(string_agg(s, ',' ORDER BY s)), '') AS checksum "
              "FROM (SELECT rr::VARCHAR AS s FROM _res rr) q;\n"
              "SELECT coalesce(sum(llm_calls),0)::BIGINT AS calls, coalesce(sum(embed_calls),0)::BIGINT AS emb "
              "FROM ai_usage();\n")
    t0 = time.perf_counter()
    rows = b._duck(script, env)
    wall = time.perf_counter() - t0
    res = next(r for r in rows if "n" in r)
    u = next(r for r in rows if "calls" in r)
    return {"surviving": int(res["n"]), "checksum": res["checksum"],
            "chat_calls": int(u["calls"]) - int(u["emb"]), "latency_s": round(wall, 1)}


# The three configs. YANN alone reduces the call COUNT but the reduced calls serialize per-chunk behind the
# join; +DEDUP buffers the reduced set and fires them in ONE concurrent batch (fixes the latency).
OFF_ENV = {"DUCKDB_YANNAKAKIS": "off"}
YANN_ENV = {"DUCKDB_YANNAKAKIS": "1"}
DEDUP_ENV = {"DUCKDB_YANNAKAKIS": "1", "DUCKDB_AI_DEDUP": "1"}


def main():
    if not os.path.exists(b.BIN):
        sys.exit(f"duckdb not found at {b.BIN}")
    ndocs = b._duck(SCHEMA + "SELECT count(*) AS n FROM docs;")[-1]["n"]
    print(f"Yannakakis bench: docs={ndocs}  gpt-5-mini  (chat_calls = docs the ai_filter evaluated)\n")
    print(f"{'query':13} {'surv':>4} | {'calls OFF':>9} {'calls ON':>8} {'reduce':>7} | "
          f"{'lOFF':>5} {'lYANN':>5} {'l+DED':>5} | match")
    print("-" * 82)
    results = {}
    for name, q in QUERIES.items():
        off = run(q, OFF_ENV)
        on = run(q, YANN_ENV)
        ded = run(q, DEDUP_ENV)
        red = (off["chat_calls"] - on["chat_calls"]) / off["chat_calls"] * 100 if off["chat_calls"] else 0
        match = off["checksum"] == on["checksum"] == ded["checksum"]
        results[name] = {"off": off, "on": on, "dedup": ded, "match": match}
        print(f"{name:13} {off['surviving']:4} | {off['chat_calls']:9} {on['chat_calls']:8} {red:6.0f}% | "
              f"{off['latency_s']:5} {on['latency_s']:5} {ded['latency_s']:5} | {match}")
    tot_off = sum(r["off"]["chat_calls"] for r in results.values())
    tot_on = sum(r["on"]["chat_calls"] for r in results.values())
    lat_on = sum(r["on"]["latency_s"] for r in results.values())
    lat_ded = sum(r["dedup"]["latency_s"] for r in results.values())
    allmatch = all(r["match"] for r in results.values())
    print("-" * 82)
    print(f"TOTAL calls {tot_off} -> {tot_on} ({(tot_off-tot_on)/tot_off*100 if tot_off else 0:+.0f}%)  |  "
          f"total latency YANN {lat_on:.0f}s -> YANN+DEDUP {lat_ded:.0f}s (concurrent-batch fix)  |  "
          f"all result-preserving: {allmatch}")
    json.dump({"docs": ndocs, "results": results}, open(f"{HERE}/yannakakis_bench_results.json", "w"), indent=1)
    if not allmatch:
        sys.exit("!! a query changed results -- reduction is NOT answer-preserving")


if __name__ == "__main__":
    main()
