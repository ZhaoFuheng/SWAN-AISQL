#!/usr/bin/env python3
"""Stage 3 scan-level factorization (DUCKDB_AI_SCAN_REGION, default ON) correctness bench.

Every AI function consumes the factorized AISQLMapData currency: fold the input into DISTINCT reps (a base
table can hold many duplicates), evaluate once per rep in waves at the cardinality floor (5*AI_MAX_CONCURRENCY),
fan the result back out. Before Stage 3 this only happened above joins; the plain scalar path deduped only
WITHIN each 2048-row chunk (per-chunk batches + the global cache), firing per-chunk waves that under-saturate
LLM concurrency on duplicated or multi-chunk inputs.

Table: N rows over D distinct docs (heavy duplication, multiple chunks). A/B on IDENTICAL SQL:
  off : DUCKDB_AI_SCAN_REGION=off  (the pre-Stage-3 scalar path)
  on  : default env                (scan regions on by default)
Assertions per shape: identical result multiset + row count; call count on <= off (factorization must never
ADD calls; typically equal -- the cache already deduped repeats); the region appears exactly where expected;
and for factorized shapes the [stream-dedup] line shows distinct==D << rows==N with floor-sized waves.

Call-safety controls (region must NOT appear, even with the flag on):
  mixed_filter   : WHERE (rn%2)=0 AND ai_filter(..)  -- mixed relational+AI filter keeps native short-circuit
  lazy_else      : CASE WHEN rn%2=0 THEN .. ELSE ai_complete(..) -- lazy ELSE runs on a row subset natively
  limit_defer    : .. LIMIT 5                        -- scalar under a constant LIMIT streams to the limit

Env: SR_N (rows, default 10000)  SR_D (distinct, default 200)  BENCH_PROXY  BENCH_EMBED  BENCH_MODEL
BENCH_CONCURRENCY.
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
    "AI_SPECULATIVE": "off",
    "AI_DEDUP_DEBUG": "1",
    "DUCKDB_AI_LIMIT": "off",  # this bench isolates scan regions; the (default-on) limit pass would early-stop the defer shapes

    "DUCKDB_SEMANTIC_PULLUP": "1",
    "DUCKDB_AI_REORDER": "1",
}
N = int(os.environ.get("SR_N", "10000"))
D = int(os.environ.get("SR_D", "200"))
SETUP = f"CREATE TABLE t AS SELECT ('doc number ' || (i%{D})) AS x, i AS rn FROM range({N}) s(i);\n"
CATS = "['health','security','economy','other']"

# (name, sql, expect_region, check_fold) -- expect_region: with the default env the plan must contain an
# Ai Region (the off config must never); check_fold: the region folds the FULL table (distinct==D, rows==N).
# limit_below: DuckDB pushes the LIMIT below the projection, so the region wraps only the post-limit rows --
# region present but folding ~5 rows, still call-safe (never more than native). limit_filter_defer: the LIMIT
# cannot sink below a WHERE, so the scalar deferral keeps native streaming (no region).
SHAPES = [
    ("select_classify",
     f"SELECT rn, ai_classify(x, {CATS}) AS c FROM t", True, True),
    ("where_filter",
     "SELECT rn, x AS c FROM t WHERE ai_filter('Is this interesting: ' || x)", True, True),
    ("case_first_when",
     "SELECT rn, CASE WHEN ai_filter('Is this interesting: ' || x) THEN 'A' ELSE 'B' END AS c FROM t", True,
     True),
    ("mixed_filter",
     "SELECT rn, x AS c FROM t WHERE (rn % 2) = 0 AND ai_filter('Is this interesting: ' || x)", False, False),
    ("lazy_else",
     "SELECT rn, CASE WHEN rn % 2 = 0 THEN 'A' ELSE ai_complete('echo ' || x) END AS c FROM t", False, False),
    ("limit_below",
     f"SELECT rn, ai_classify(x, {CATS}) AS c FROM t LIMIT 5", True, False),
    ("limit_filter_defer",
     "SELECT rn, x AS c FROM t WHERE ai_filter('Is this interesting: ' || x) LIMIT 5", False, False),
]

FLAGS = ("DUCKDB_AI_DEDUP", "DUCKDB_AI_LIMIT", "DUCKDB_AI_REORDER", "DUCKDB_SEMANTIC_PULLUP",
         "DUCKDB_AI_GROUP_JOIN", "DUCKDB_AI_SCAN_REGION", "DUCKDB_YANNAKAKIS")
DBG = re.compile(
    r"\[stream-dedup\] distinct=(\d+) rows=(\d+) waves=(\d+) first_wave_at_rows=(\d+) passed=(-?\d+) "
    r"limit=(-?\d+) floor=(\d+) min_full_wave=(\d+) last_flush=(\d+)")


def duck(script, extra=None, plain=False):
    env = dict(os.environ)
    for k in FLAGS:
        env.pop(k, None)
    env.update(ENV_COMMON)
    if extra:
        env.update(extra)
    args = [BIN, "-c", script] if plain else [BIN, "-noheader", "-list", "-c", script]
    p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=3600)
    if not plain and p.returncode != 0:
        raise RuntimeError(f"duckdb failed: {p.stderr[:800]}")
    return [l for l in p.stdout.strip().splitlines() if l != ""], p.stderr


def measure(sql, extra=None):
    script = (f"{SETUP}CREATE TEMP TABLE r AS {sql};\n"
              "SELECT coalesce(string_agg(s, '|' ORDER BY s), '') FROM "
              "(SELECT rn::VARCHAR||'/'||coalesce(c::VARCHAR,'N') AS s FROM r) q;\n"
              "SELECT count(*) FROM r;\n"
              "SELECT coalesce(sum(llm_calls),0)-coalesce(sum(embed_calls),0) FROM ai_usage();\n")
    rows, err = duck(script, extra)
    m = DBG.search(err)
    dbg = None
    if m:
        dbg = {"distinct": int(m.group(1)), "rows": int(m.group(2)), "waves": int(m.group(3)),
               "floor": int(m.group(7)), "min_full_wave": int(m.group(8)), "last_flush": int(m.group(9))}
    return {"sig": rows[0], "n": int(rows[1]), "calls": int(rows[2]), "dbg": dbg}


def has_region(sql, extra=None):
    out, err = duck(f"{SETUP}PRAGMA explain_output='OPTIMIZED_ONLY'; EXPLAIN {sql};", extra, plain=True)
    return any("Ai Region" in l for l in out) or "Ai Region" in err


def run_shape(name, sql, expect_region, check_fold):
    off = measure(sql, {"DUCKDB_AI_SCAN_REGION": "off"})
    on = measure(sql)  # default env: scan regions ON by default
    reg_off = has_region(sql, {"DUCKDB_AI_SCAN_REGION": "off"})
    reg_on = has_region(sql)
    checks = {
        "sig": on["sig"] == off["sig"] and on["n"] == off["n"],
        "calls": on["calls"] <= off["calls"],
        "region": (reg_on == expect_region) and not reg_off,
    }
    if check_fold:
        d = on["dbg"]
        checks["fold"] = bool(d) and d["distinct"] == D and d["rows"] == N
        checks["floor"] = bool(d) and (d["min_full_wave"] == 0 or d["min_full_wave"] >= d["floor"])
    ok = all(checks.values())
    return {"name": name, "off": off, "on": on, "reg_on": reg_on, "checks": checks, "ok": ok}


def main():
    if not os.path.exists(BIN):
        sys.exit(f"duckdb not found at {BIN}")
    sys.stderr.write(f"N={N} D={D} (dup x{N // D}) shapes={[s[0] for s in SHAPES]}\n")
    results = [run_shape(*s) for s in SHAPES]
    print(f"\n=== Stage 3 scan-level factorization (DUCKDB_AI_SCAN_REGION off vs default-on; N={N} D={D}) ===")
    print(f"{'shape':<18}{'n':<8}{'calls off/on':<14}{'region':<8}{'distinct/rows':<15}{'waves':<7}{'ok'}")
    print("-" * 78)
    bad = []
    for r in results:
        if not r["ok"]:
            bad.append((r["name"], r["checks"]))
        d = r["on"]["dbg"] or {}
        dr = f"{d.get('distinct','-')}/{d.get('rows','-')}" if d else "-"
        print(f"{r['name']:<18}{r['off']['n']:<8}{str(r['off']['calls'])+'/'+str(r['on']['calls']):<14}"
              f"{str(r['reg_on']):<8}{dr:<15}{d.get('waves','-'):<7}{r['ok']}")
    print("\ndistinct/rows: the region folded N duplicated rows to D distinct reps (global, cross-chunk dedup)")
    print("before any LLM call, then fanned the per-rep result back out. Controls (region=False) prove the")
    print("call-safety guards: mixed filters, lazy CASE branches, and LIMIT queries keep native execution.")
    if bad:
        print(f"\n!!! {len(bad)} FAILED: {[b[0] for b in bad]}")
        for name, checks in bad:
            print(f"    {name}: {checks}")
        sys.exit(1)
    print("\nAll shapes green: factorized == scalar results, never more calls, regions exactly where expected.")


if __name__ == "__main__":
    main()
