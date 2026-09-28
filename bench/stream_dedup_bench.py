#!/usr/bin/env python3
"""Does STREAMING the AI-dedup operator overlap LLM calls with input (and early-tear-down under a LIMIT)?

The AI-dedup operator used to be a pipeline breaker: buffer the WHOLE join output, then fire every distinct
LLM call once (first call waits for full materialization). Streaming (always; the scalar path is the reference) dedups into an AISQLMapChunk as rows arrive and fires the LLM calls in WAVES once
~5 x concurrency new distinct keys accumulate -- overlapping AI latency with the join -- and, under a
pushed LIMIT, returns FINISHED once the passing outputs reach k (so the join's tail is never materialized).

Two shapes where the AI fires IN the dedup (genuine fan-out above the join):
  node_limit    : a multi-table ai_filter in the JOIN CONDITION (pulled up into an ai_function_with_embed node
                  above the fan-out self-join) + a small LIMIT k. Streaming should EARLY-TEAR-DOWN: rows
                  buffered << full join output, calls ~= blocking (both count-weighted early-stop), lower wall.
  scalar_nolimit: an ai_classify in a PROJECTION above the fan-out self-join, NO limit. Streaming is
                  overlap-only: SAME calls + identical rows, but the FIRST wave fires after ~WAVE distinct
                  keys (first_wave_at_rows << full), proving the overlap deterministically.

blocking vs streaming on the same build. Metrics: chat_calls = llm_calls - embed_calls via ai_usage(); the
[stream-dedup] AI_DEDUP_DEBUG stderr line gives distinct / rows / waves / first_wave_at_rows / floor /
min_full_wave / last_flush. Correctness per shape: node_limit -> res subset of full AND n == min(k, full_passing);
scalar_nolimit -> streaming rows == blocking rows (identical). PLUS the Stage-1 two-currency FLOOR INVARIANT
(floor_ok): every floor-triggered (Sink) AISQLMapData wave carries >= floor = 5*AI_MAX_CONCURRENCY distinct reps
(min_full_wave >= floor), so each LLM batch saturates concurrency; only the final Finalize flush (last_flush) may
be smaller. A violation fails the bench. Env: SD_DOCS (default 200)  BENCH_PROXY  BENCH_EMBED  BENCH_MODEL
BENCH_CONCURRENCY.
"""
import json
import os
import re
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
    "AI_SPECULATIVE": "off",  # pure pull-up (no speculative leaf) so the AI node sits above the join
    "AI_DEDUP_DEBUG": "1",            # emit the [stream-dedup] diagnostics line
}
DOCS = int(os.environ.get("SD_DOCS", "200"))
LOAD = (f"CREATE TABLE gov AS SELECT row_number() OVER () AS rn, id, summary, "
        f"(row_number() OVER ())::INT % 20 AS jk FROM read_csv('{HERE}/govreport_summary.csv', header=true) "
        f"LIMIT {DOCS};\n")

# TWO multi-table ai_filters -> the reorder builds ONE ai_function_with_embed NODE; being multi-table it can't
# be pushed below the join, so it sits above the fan-out join and the dedup wraps it (+ carries k under a
# LIMIT). Loose (mostly-true) predicates so >= k pairs pass EARLY -> the LIMIT early-tears-down after ~1 wave.
HAB1 = ("ai_filter('Do both of these texts appear to be government or policy documents? ' || "
        "substr(a.summary,1,200) || ' ||| ' || substr(b.summary,1,200))")
HAB2 = ("ai_filter('Are both of these texts written in English? ' || substr(a.summary,1,200) || ' ||| ' || "
        "substr(b.summary,1,200))")
C = "ai_classify(a.summary, ['health', 'security', 'economy', 'other'])"

# Fan-out multiplier: broadcast each DISTINCT doc to FAN copies via a cross join. DOCS*FAN rows. With
# DOCS=963, FAN=3 -> 2889 rows > 2048 (STANDARD_VECTOR_SIZE) so the join output spans MULTIPLE DataChunks and
# the streaming dedup fires several waves as chunks arrive (blocking buffers all chunks, fires once in Finalize).
FAN = int(os.environ.get("SD_FAN", "3"))
XJOIN = f"CROSS JOIN (SELECT * FROM range({FAN})) t"
# TWO single-table ai_filters -> reorder builds a NODE (a lone scalar ai_filter under a LIMIT is left to stream,
# not deduped); pull-up lifts it above the cross join; the dedup wraps it and carries k for the count-weighted
# early-stop. Loose predicates so passers appear early.
HF = ("ai_filter('Is this a government or policy document? ' || substr(a.summary,1,300)) AND "
      "ai_filter('Is this document written in English? ' || substr(a.summary,1,300))")

# MULTI-WAVE currency shape: MW_N DISTINCT synthetic docs (no fan-out, so distinct reps arrive in SCAN order
# spread across chunks) above a trivial cross product. With MW_N=4150 and floor=100 (concurrency 20) the AI
# region sees 4150 distinct reps in chunks of 2048/2048/54 -> TWO floor-triggered Sink waves (each >= floor) +
# one Finalize flush of 54 (< floor). This is the ONLY shape that exercises the floor across MULTIPLE full waves
# + a sub-floor tail flush; the gov self-join / fan-out shapes all fit their distinct set in the first chunk
# (one wave). Uses its own subquery table, not `gov`.
MW_N = int(os.environ.get("SD_MW_N", "4150"))
MW_SRC = f"(SELECT 'doc number ' || i AS summary FROM range({MW_N}) t(i))"

# (name, kind, body_without_limit, k)  k = -1 means no LIMIT
QUERIES = [
    # AI NODE fires IN the dedup over DISTINCT pairs (>> WAVE); LIMIT -> streaming early-tears-down the join.
    ("node_limit", "NODE_LIMIT",
     f"SELECT a.id AS aid, b.id AS bid FROM gov a JOIN gov b ON a.jk = b.jk WHERE a.id < b.id AND {HAB1} AND {HAB2}",
     5),
    # scalar ai_classify deduped above the fan-out self-join, no limit -> overlap only (same calls, first wave early).
    ("scalar_nolimit", "SCALAR_NOLIMIT",
     f"SELECT a.id AS aid, {C} AS c FROM gov a JOIN gov b ON a.jk = b.jk", -1),
    # USER SHAPE: DOCS distinct docs each broadcast to FAN copies (DOCS*FAN rows, crosses the chunk boundary),
    # ai_classify deduped -> DOCS calls (blocking == streaming). Streaming fires waves across chunks; blocking
    # fires one Finalize batch. no-limit -> shows dedup call-collapse + multi-wave, same total calls.
    ("dup_fanout", "DUP_FANOUT",
     f"SELECT a.id AS aid, {C} AS c FROM gov a {XJOIN}", -1),
    # High fan-out (FAN copies/doc) node ai_filter above the cross join, pull-up keeps it above + LIMIT k. One
    # passing doc broadcasts to FAN output rows, so k is met after ~1 passer -> streaming returns FINISHED after
    # ~chunk 1 and never scans/buffers the tail; blocking materializes ALL DOCS*FAN rows first. Biggest margin.
    ("fanout_limit", "FANOUT_LIMIT",
     f"SELECT a.id AS aid FROM gov a {XJOIN} WHERE {HF}", 5),
    # MULTI-WAVE: MW_N distinct reps across chunks -> multiple floor-triggered waves + a sub-floor flush. main()
    # additionally asserts streaming waves >= 2 for this shape (else the multi-wave floor path went untested).
    ("multiwave", "MULTIWAVE",
     f"SELECT a.summary AS aid, {C} AS c FROM {MW_SRC} a CROSS JOIN (SELECT * FROM range(1)) t", -1),
]
ONLY = set(s for s in os.environ.get("SD_ONLY", "").split(",") if s)

# The region always streams. The "blocking" reference evaluates the node per chunk on the scalar path (no
# region); pull-up + reorder always on so the AI node is above the join.
CFGS = {
    "blocking":  {"DUCKDB_SEMANTIC_PULLUP": "1", "DUCKDB_AI_REORDER": "1", "DUCKDB_AI_DEDUP": "off"},
    "streaming": {"DUCKDB_SEMANTIC_PULLUP": "1", "DUCKDB_AI_REORDER": "1"},
}
FLAGS = ("DUCKDB_AI_DEDUP", "DUCKDB_AI_LIMIT", "DUCKDB_AI_REORDER", "DUCKDB_SEMANTIC_PULLUP")


def _duck(script, env_extra):
    env = dict(os.environ)
    env.update(ENV_COMMON)
    for k in FLAGS:
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
    rows = [r for arr in objs for r in (arr if isinstance(arr, list) else [arr])]
    return rows, p.stderr


def overhead():
    t0 = time.perf_counter()
    _duck(f"{LOAD} SELECT count(*) AS n FROM gov;", CFGS["blocking"])
    return time.perf_counter() - t0


def full_count(body):
    rows, _ = _duck(f"{LOAD}CREATE TEMP TABLE fullr AS {body};\nSELECT count(*) AS n FROM fullr;\n", CFGS["blocking"])
    return int(next(r for r in rows if "n" in r)["n"])


DBG = re.compile(
    r"\[stream-dedup\] distinct=(\d+) rows=(\d+) waves=(\d+) first_wave_at_rows=(\d+) passed=(-?\d+) limit=(-?\d+) "
    r"floor=(\d+) min_full_wave=(\d+) last_flush=(\d+)")


def parse_dbg(stderr):
    # The `res` (LIMIT) run is created FIRST, so its [stream-dedup] line comes before the un-limited `fullr`
    # run's -- take the first, which reflects the early-teardown/overlap of the measured query.
    m = DBG.search(stderr)
    if not m:
        return None
    return {"distinct": int(m.group(1)), "rows": int(m.group(2)), "waves": int(m.group(3)), "first": int(m.group(4)),
            "passed": int(m.group(5)), "limit": int(m.group(6)), "floor": int(m.group(7)),
            "min_full_wave": int(m.group(8)), "last_flush": int(m.group(9))}


def floor_ok(dbg):
    """Stage-1 cardinality-floor invariant: every floor-triggered (Sink) wave carries >= floor distinct reps;
    only the final Finalize flush (the leftover tail) may be smaller. Vacuously true when the whole input has
    fewer distinct reps than the floor (one flush wave, no full wave) or under an early LIMIT teardown."""
    if not dbg:
        return True, "no [stream-dedup] line (blocking path or AI fired elsewhere)"
    floor = dbg["floor"]
    mfw = dbg["min_full_wave"]
    if mfw == 0:
        # No floor-triggered wave fired -- valid iff there weren't >= floor distinct reps to trigger one, OR the
        # run tore down early under a LIMIT before accumulating a full wave.
        if dbg["distinct"] < floor or (dbg["limit"] >= 0 and dbg["passed"] >= dbg["limit"]):
            return True, f"no full wave (distinct={dbg['distinct']} < floor={floor} or early LIMIT teardown)"
        return False, f"distinct={dbg['distinct']} >= floor={floor} but NO floor-triggered wave fired"
    if mfw < floor:
        return False, f"min_full_wave={mfw} < floor={floor} (a Sink wave under-saturated concurrency)"
    return True, f"min_full_wave={mfw} >= floor={floor}"


def run(body, k, cfg_env, full_n, oh):
    limited = f" LIMIT {k}" if k >= 0 else ""
    sig_expr = "aid::VARCHAR"  # row signature over whatever columns res actually has
    if "bid" in body:
        sig_expr += "||'/'||bid::VARCHAR"
    elif "AS c" in body:
        sig_expr += "||'/'||c"
    stmts = [
        f"{LOAD}CREATE TEMP TABLE res AS {body}{limited};",
        "SELECT count(*) AS n FROM res;",
        "SELECT coalesce(sum(llm_calls),0) AS llm, coalesce(sum(embed_calls),0) AS emb FROM ai_usage();",
        f"CREATE TEMP TABLE fullr AS {body};",
        "SELECT count(*) AS unsound FROM (SELECT * FROM res EXCEPT SELECT * FROM fullr);",
        f"SELECT coalesce(string_agg(x, '|'), '') AS sig FROM (SELECT ({sig_expr}) AS x FROM res ORDER BY 1);",
    ]
    t0 = time.perf_counter()
    rows, stderr = _duck("\n".join(stmts) + "\n", cfg_env)
    wall = time.perf_counter() - t0
    n = int(next(r for r in rows if "n" in r)["n"])
    u = next(r for r in rows if "llm" in r)
    unsound = int(next(r for r in rows if "unsound" in r)["unsound"])
    sig = next(r for r in rows if "sig" in r)["sig"]
    dbg = parse_dbg(stderr)
    ok = unsound == 0 and (n == min(k, full_n) if k >= 0 else n == full_n)
    return {"n": n, "calls": int(u["llm"]) - int(u["emb"]), "lat": max(0.0, wall - oh), "ok": ok, "sig": sig,
            "dbg": dbg}


def main():
    if not os.path.exists(BIN):
        sys.exit(f"duckdb not found at {BIN}")
    queries = [q for q in QUERIES if not ONLY or q[0] in ONLY]
    sys.stderr.write(f"docs={DOCS} fan={FAN} shapes={[q[0] for q in queries]}\n")
    oh = overhead()
    results, bad = [], []
    for name, kind, body, k in queries:
        fn = full_count(body)
        r = {cfg: run(body, k, env, fn, oh) for cfg, env in CFGS.items()}
        # scalar_nolimit must be result-IDENTICAL streaming vs blocking; node_limit must be a valid k-subset.
        preserved = (r["streaming"]["sig"] == r["blocking"]["sig"]) if k < 0 else True
        # Stage-1 currency check: on the streaming path every floor-triggered wave saturated concurrency (>= floor).
        fok, fmsg = floor_ok(r["streaming"]["dbg"])
        # The multiwave shape exists specifically to exercise >= 2 full waves + a sub-floor flush; if it collapses
        # to one wave the multi-wave floor path went untested, which is itself a regression for this shape.
        if name == "multiwave":
            sw = (r["streaming"]["dbg"] or {}).get("waves", 0)
            if sw < 2:
                fok, fmsg = False, f"multiwave fired only {sw} wave(s) (expected >= 2; raise SD_MW_N?)"
        results.append((name, kind, k, fn, r, preserved, fok))
        if not (r["blocking"]["ok"] and r["streaming"]["ok"] and preserved and fok):
            bad.append(name)
        db, ds = r["blocking"]["dbg"], r["streaming"]["dbg"]
        sys.stderr.write(f"[{name}] full={fn}  calls blk {r['blocking']['calls']} / str {r['streaming']['calls']}  "
                         f"str_rows_buffered={ds['rows'] if ds else '?'}  str_first_wave_at={ds['first'] if ds else '?'}"
                         f"  waves={ds['waves'] if ds else '?'}  ok={r['blocking']['ok'] and r['streaming']['ok']}"
                         f" preserved={preserved}  floor[{fmsg}]={fok}\n")
        sys.stderr.flush()
    report(results)
    if bad:
        print(f"\n!!! {len(bad)} SHAPES FAILED CORRECTNESS: {bad}")
        sys.exit(1)


def report(results):
    print(f"\n=== Streaming AI-dedup: overlap + LIMIT early-teardown + Stage-1 currency floor (docs={DOCS}) ===")
    print(f"{'shape':<16}{'k/full':<10}{'blk_calls':<10}{'str_calls':<10}{'str_rows':<9}{'first_wave':<11}"
          f"{'waves':<6}{'floor':<7}{'minwave':<9}{'flush':<7}{'blk_lat':<9}{'str_lat':<9}{'ok'}")
    print("-" * 118)
    for name, kind, k, fn, r, preserved, fok in results:
        ds = r["streaming"]["dbg"] or {}
        note = ""
        if k >= 0 and ds:
            note = "  <- teardown" if ds.get("rows", fn) < fn else ""
        elif ds:
            note = "  <- overlap" if ds.get("first", fn) < fn else ""
        allok = r['blocking']['ok'] and r['streaming']['ok'] and preserved and fok
        print(f"{name:<16}{str(k)+'/'+str(fn):<10}{r['blocking']['calls']:<10}{r['streaming']['calls']:<10}"
              f"{ds.get('rows','?'):<9}{ds.get('first','?'):<11}{ds.get('waves','?'):<6}{ds.get('floor','?'):<7}"
              f"{ds.get('min_full_wave','?'):<9}{ds.get('last_flush','?'):<7}"
              f"{r['blocking']['lat']:<9.2f}{r['streaming']['lat']:<9.2f}{allok}{note}")
    print("\nnode_limit: str_rows << full = early teardown (join tail never materialized).")
    print("scalar_nolimit: first_wave << full = LLM calls overlap the join (same total calls, identical rows).")
    print("floor/minwave: Stage-1 currency invariant -- every floor-triggered wave carries >= floor distinct reps")
    print("               (minwave >= floor); flush = the final leftover-tail wave (may be < floor). minwave=0 =")
    print("               no full wave fired (distinct < floor or early LIMIT teardown).")


if __name__ == "__main__":
    main()
