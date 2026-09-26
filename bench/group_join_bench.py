#!/usr/bin/env python3
"""Stage 2 group-join (AIGroupJoinRewrite, DUCKDB_AI_GROUP_JOIN) correctness + materialization bench.

Stage 1 places the AI region ABOVE the join: the join builds the full N*M fan-out and the region SINKS all of
it (a blocking buffer of every fan-out row) just to dedup the AI call down to the distinct inputs. The Stage-2
pass pushes a single-side-key region BELOW the join onto the key-owning side -- which the (purely relational,
AI-free) Yannakakis semi-join reduction has already restricted to the keys that survive the join:

    Region[ Join(L, R) ]   (AI key ⊆ L)   ->   Join( Region[L_semi_reduced], R )

Same results (the join fans out the appended result column), same AI calls (both dedup to the surviving
distinct keys), but the region buffers |L| rows instead of the whole fan-out -- the fan-out is never
materialized to feed the AI.

This bench A/Bs the pass on IDENTICAL SQL: DUCKDB_AI_GROUP_JOIN off (Stage 1) vs on (Stage 2). Per shape it
asserts, on the same build + mock backend:
  (a) result multiset identical (order-insensitive signature),
  (b) identical AI call count,
  (c) plan position: the region moves below the INNER/CROSS join when the pass should fire (expect_push),
      and stays above for the multi-side-key control (the AI call reads both sides -> no legal push),
  (d) materialization: region-buffered rows shrink (from the [stream-dedup] AI_DEDUP_DEBUG line) when pushed.
A hand-written semi-join group-join for proj_classify is kept as an independent reference (same signature).

Env: GJ_MULT (table-size multiplier)  BENCH_PROXY  BENCH_EMBED  BENCH_MODEL  BENCH_CONCURRENCY.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))  # repo root (bench/ is one level down)
BIN = os.environ.get("DUCKDB_BIN", os.path.join(ROOT, "build/release/duckdb"))

ENV_COMMON = {
    "AI_PROXY_URL": os.environ.get("BENCH_PROXY", "http://localhost:4009"),
    "AI_MODEL": os.environ.get("BENCH_MODEL", "gpt-5-mini"),
    "AI_EMBED_URL": os.environ.get("BENCH_EMBED", "http://localhost:4009"),
    "AI_EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "AI_MAX_CONCURRENCY": os.environ.get("BENCH_CONCURRENCY", "20"),
    "AI_SPECULATIVE_THRESHOLD": "0",  # no speculative leaf -> the AI region forms cleanly above the join
    "AI_DEDUP_DEBUG": "1",            # emit the [stream-dedup] line (region-buffered rows = materialization)
    "DUCKDB_SEMANTIC_PULLUP": "1",
    "DUCKDB_AI_REORDER": "1",
}
MULT = int(os.environ.get("GJ_MULT", "1"))
NA, NB, NK_A, NK_B = 40 * MULT, 30 * MULT, 7, 5
SETUP = (f"CREATE TABLE a AS SELECT ('topic ' || (i%{NK_A})) AS x, i AS aid FROM range({NA}) t(i);\n"
         f"CREATE TABLE b AS SELECT ('topic ' || (i%{NK_B})) AS y, i AS bid FROM range({NB}) t(i);\n")
CATS = "['health','security','economy','other']"

# (name, sql, join_marker, expect_push)
#   join_marker: the plan line identifying the fan-out join (position of "Ai Region" relative to it = the check)
#   expect_push: True -> the pass must move the region below the join; False -> it must leave it above.
SHAPES = [
    ("proj_classify",
     f"SELECT a.x AS x, ai_classify(a.x, {CATS}) AS c FROM a JOIN b ON a.x = b.y",
     "Join Type: INNER", True),
    ("proj_score",
     "SELECT a.x AS x, ai_score(a.x, 'how interesting is this topic')::VARCHAR AS c FROM a JOIN b ON a.x = b.y",
     "Join Type: INNER", True),
    # cross product: every side row reaches the output, so the push needs no semi-reduction gate.
    ("proj_classify_cross",
     f"SELECT a.x AS x, ai_classify(a.x, {CATS}) AS c FROM a CROSS JOIN (SELECT * FROM range(3)) t",
     "Cross Product", True),
    # CONTROL: the AI call reads BOTH sides -> no single key side exists; the pass must not fire. NB: the
    # right-side column must NOT be the join key -- `a.x || b.y` under `ON a.x = b.y` is canonicalized to a
    # single-side input by the equi-join equivalence (the pass then legally pushes it; observed, results
    # identical). b.bid is not equated to anything on the left, so this stays genuinely two-sided.
    ("multi_side_control",
     f"SELECT a.x AS x, ai_classify(a.x || ' / ' || b.bid::VARCHAR, {CATS}) AS c FROM a JOIN b ON a.x = b.y",
     "Join Type: INNER", False),
]

# Independent reference: the hand-written semi-join group-join for proj_classify (validated result-identical
# before the pass existed). The AI stays in a projection over the DISTINCT SURVIVING keys.
HAND_CLASSIFY = (f"WITH surv AS (SELECT DISTINCT a.x AS k FROM a SEMI JOIN b ON a.x = b.y), "
                 f"m AS (SELECT k, ai_classify(k, {CATS}) AS c FROM surv) "
                 f"SELECT j.x AS x, m.c AS c FROM (SELECT a.x FROM a JOIN b ON a.x = b.y) j JOIN m ON j.x = m.k")

FLAGS = ("DUCKDB_AI_DEDUP", "DUCKDB_AI_LIMIT", "DUCKDB_AI_REORDER", "DUCKDB_SEMANTIC_PULLUP",
         "DUCKDB_AI_STREAM_DEDUP", "DUCKDB_AI_GROUP_JOIN", "DUCKDB_YANNAKAKIS")
DBG = re.compile(r"\[stream-dedup\] distinct=(\d+) rows=(\d+)")


def duck(script, extra=None):
    env = dict(os.environ)
    for k in FLAGS:
        env.pop(k, None)
    env.update(ENV_COMMON)
    if extra:
        env.update(extra)
    p = subprocess.run([BIN, "-noheader", "-list", "-c", script], env=env, capture_output=True, text=True,
                       timeout=3600)
    if p.returncode != 0:
        raise RuntimeError(f"duckdb failed: {p.stderr[:800]}")
    return [l for l in p.stdout.strip().splitlines() if l != ""], p.stderr


def measure(sql, extra=None):
    # sig = order-insensitive multiset signature over the WHOLE row (struct-cast, column-agnostic);
    # calls = chat calls; regbuf = rows the region sank.
    script = (f"{SETUP}CREATE TEMP TABLE r AS {sql};\n"
              "SELECT coalesce(string_agg(s, '|' ORDER BY s), '') FROM (SELECT r::VARCHAR AS s FROM r) q;\n"
              "SELECT count(*) FROM r;\n"
              "SELECT coalesce(sum(llm_calls),0)-coalesce(sum(embed_calls),0) FROM ai_usage();\n")
    rows, err = duck(script, extra)
    regbuf = sum(int(m.group(2)) for m in DBG.finditer(err))
    return {"sig": rows[0], "n": int(rows[1]), "calls": int(rows[2]), "regbuf": regbuf}


def plan_pos(sql, join_marker, extra=None):
    script = f"{SETUP}PRAGMA explain_output='OPTIMIZED_ONLY'; EXPLAIN {sql};"
    p_env = dict(os.environ)
    for k in FLAGS:
        p_env.pop(k, None)
    p_env.update(ENV_COMMON)
    if extra:
        p_env.update(extra)
    p = subprocess.run([BIN, "-c", script], env=p_env, capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    region_line = join_line = None
    for i, line in enumerate(out.splitlines()):
        if region_line is None and "Ai Region" in line:
            region_line = i
        if join_line is None and join_marker in line:
            join_line = i
    if region_line is None or join_line is None:
        return "missing"
    return "above" if region_line < join_line else "below"


# DISTINCT / duplicate-insensitive aggregate consumers: the pass converts the INNER join to SEMI (the fan-out's
# multiplicity is irrelevant to the consumer, so the join's only role is existence) -- the expand disappears
# entirely: each kept-side row flows once, the region folds survivors. count(*) is duplicate-SENSITIVE -> the
# INNER must remain (control). (name, sql, expect_convert)
SEMI_SHAPES = [
    ("distinct_classify",
     f"SELECT DISTINCT ai_classify(a.x, {CATS}) AS c FROM a JOIN b ON a.x = b.y", True),
    ("group_min",
     f"SELECT ai_classify(a.x, {CATS}) AS c, min(a.aid) AS m FROM a JOIN b ON a.x = b.y GROUP BY c", True),
    ("count_star_control",
     f"SELECT ai_classify(a.x, {CATS}) AS c, count(*) AS n FROM a JOIN b ON a.x = b.y GROUP BY c", False),
]


def inner_present(sql, extra=None):
    script = f"{SETUP}PRAGMA explain_output='OPTIMIZED_ONLY'; EXPLAIN {sql};"
    env = dict(os.environ)
    for k in FLAGS:
        env.pop(k, None)
    env.update(ENV_COMMON)
    if extra:
        env.update(extra)
    p = subprocess.run([BIN, "-c", script], env=env, capture_output=True, text=True, timeout=600)
    return "Join Type: INNER" in (p.stdout + p.stderr)


def run_semi_shape(name, sql, expect_convert):
    off = measure(sql, {"DUCKDB_AI_GROUP_JOIN": "off"})
    on = measure(sql, {"DUCKDB_AI_GROUP_JOIN": "1"})
    inner_off = inner_present(sql, {"DUCKDB_AI_GROUP_JOIN": "off"})
    inner_on = inner_present(sql, {"DUCKDB_AI_GROUP_JOIN": "1"})
    checks = {
        "sig": on["sig"] == off["sig"] and on["n"] == off["n"],
        "calls": on["calls"] <= off["calls"],
        "inner": inner_off and (inner_on != expect_convert),
    }
    if expect_convert and off["regbuf"] > 0:
        # A region existed: it must now fold the survivors, not the fan-out. (GROUP BY ai_classify(..) holds
        # the AI in the aggregate's group exprs -- no region -- and its win is rows-into-the-aggregate.)
        checks["regbuf"] = on["regbuf"] < off["regbuf"]
    ok = all(checks.values())
    return {"name": name, "off": off, "on": on, "inner_on": inner_on, "checks": checks, "ok": ok}


def run_shape(name, sql, join_marker, expect_push):
    off = measure(sql, {"DUCKDB_AI_GROUP_JOIN": "off"})  # Stage 1: region above the join
    on = measure(sql, {"DUCKDB_AI_GROUP_JOIN": "1"})  # Stage 2: pushed below (when legal)
    pos_off = plan_pos(sql, join_marker, {"DUCKDB_AI_GROUP_JOIN": "off"})
    pos_on = plan_pos(sql, join_marker, {"DUCKDB_AI_GROUP_JOIN": "1"})
    checks = {
        "sig": on["sig"] == off["sig"] and on["n"] == off["n"],
        "calls": on["calls"] == off["calls"],
        "pos": (pos_off == "above" and pos_on == ("below" if expect_push else "above")),
        "regbuf": (on["regbuf"] < off["regbuf"]) if expect_push else (on["regbuf"] == off["regbuf"]),
    }
    ok = all(checks.values())
    return {"name": name, "off": off, "on": on, "pos_off": pos_off, "pos_on": pos_on, "checks": checks, "ok": ok}


def main():
    if not os.path.exists(BIN):
        sys.exit(f"duckdb not found at {BIN}")
    sys.stderr.write(f"mult={MULT} NA={NA} NB={NB} shapes={[s[0] for s in SHAPES]}\n")
    results = [run_shape(*s) for s in SHAPES]
    # Independent hand-written reference for proj_classify (run under flag OFF env).
    hand = measure(HAND_CLASSIFY, {"DUCKDB_AI_GROUP_JOIN": "off"})
    ref_ok = hand["sig"] == results[0]["off"]["sig"]

    print(f"\n=== Stage 2 group-join pass (DUCKDB_AI_GROUP_JOIN off vs on, mult={MULT}) ===")
    print(f"{'shape':<20}{'n':<7}{'calls off/on':<14}{'regbuf off/on':<16}{'plan off->on':<15}{'ok'}")
    print("-" * 84)
    bad = []
    for r in results:
        if not r["ok"]:
            bad.append((r["name"], r["checks"]))
        print(f"{r['name']:<20}{r['off']['n']:<7}{str(r['off']['calls'])+'/'+str(r['on']['calls']):<14}"
              f"{str(r['off']['regbuf'])+'/'+str(r['on']['regbuf']):<16}"
              f"{r['pos_off']+'->'+r['pos_on']:<15}{r['ok']}")
    print(f"{'hand_ref(classify)':<20}{hand['n']:<7}{hand['calls']:<14}{'-':<16}{'-':<15}{ref_ok}")
    print("\nregbuf = rows the AI region sinks. Pushed below the join it buffers the (semi-reduced) side, not")
    print("the fan-out; identical results + calls, the fan-out is never materialized to feed the AI.")
    if not ref_ok:
        bad.append(("hand_ref", {"sig": False}))

    semi_results = [run_semi_shape(*s) for s in SEMI_SHAPES]
    print(f"\n=== DISTINCT / duplicate-insensitive consumers: INNER -> SEMI (skip the expand entirely) ===")
    print(f"{'shape':<20}{'n':<7}{'calls off/on':<14}{'regbuf off/on':<16}{'INNER on?':<11}{'ok'}")
    print("-" * 74)
    for r in semi_results:
        if not r["ok"]:
            bad.append((r["name"], r["checks"]))
        print(f"{r['name']:<20}{r['off']['n']:<7}{str(r['off']['calls'])+'/'+str(r['on']['calls']):<14}"
              f"{str(r['off']['regbuf'])+'/'+str(r['on']['regbuf']):<16}{str(r['inner_on']):<11}{r['ok']}")
    print("\nConverted shapes: the INNER join is GONE (SEMI only) -- no fan-out is ever produced, the region")
    print("folds survivors, and the DISTINCT/aggregate reads one row per kept-side row. count(*) is duplicate-")
    print("sensitive, so its INNER stays (control).")
    if bad:
        print(f"\n!!! {len(bad)} FAILED: {[b[0] for b in bad]}")
        for name, checks in bad:
            print(f"    {name}: {checks}")
        sys.exit(1)
    print("\nAll shapes: pass preserves results + calls; pushes exactly where legal; materialization shrinks.")


if __name__ == "__main__":
    main()
