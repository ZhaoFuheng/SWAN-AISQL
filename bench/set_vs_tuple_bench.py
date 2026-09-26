#!/usr/bin/env python3
"""Factorized SET evaluation vs TUPLE-at-a-time evaluation on a 3-table semantic join.

The same conjunctive semantic join -- unary(A) AND edge(A,B) AND edge(B,C) over a cross
product -- evaluated three ways:

  set        -- defaults: the n-ary factor graph (member/pair domains, global stage order
                smallest-live-domain-first, cascade between edges; tuples enumerated call-free)
  tuple      -- ai_join_factorize=off + speculative threshold 0: the region folds the FULL
                cross product and the folded 3-leaf node runs per-TUPLE with the DP reorder,
                short-circuit, single-flight dedup and pull-ahead (the pre-graph architecture)
  tuple+leaf -- ai_join_factorize=off with default speculative leaves: tuple-at-a-time above,
                but estimate-gated leaf nodes pre-prune the sides (the hybrid)

Axes: table size N (row currency pressure) and mock pass rate p% (selectivity: low p = strong
pruning available). [sleep=NN] gives every call a deterministic latency so wall time reflects
scheduling, not just engine CPU. Asserts identical results across modes at every point.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("DUCKDB_BIN", os.path.abspath(os.path.join(HERE, "../build/release/duckdb")))
SLEEP = int(os.environ.get("SVT_SLEEP_MS", "100"))

MODES = {
    "set": "",
    "set-lazy": "SET ai_debug_graph_eval='lazy';",
    "set-adaptive": "SET ai_debug_graph_eval='lazy-adaptive';",
    "tuple": "SET ai_join_factorize='off'; SET ai_debug_speculative_threshold=0;",
    "tuple+leaf": "SET ai_join_factorize='off';",
}


def run_mode(settings, n, p, tag):
    knob = f"[p={p}][sleep={SLEEP}]"
    out = f"/tmp/svt_{tag}.csv"
    usage = f"/tmp/svt_usage_{tag}.csv"
    sql = f"""
CALL ai_mock_start();
{settings}
CREATE TABLE ta AS SELECT 'a'||i::VARCHAR AS x FROM range({n}) r(i);
CREATE TABLE tb AS SELECT 'b'||i::VARCHAR AS y FROM range({n}) r(i);
CREATE TABLE tc AS SELECT 'c'||i::VARCHAR AS z FROM range({n}) r(i);
SELECT ai_usage_reset();
COPY (
  SELECT * FROM ta, tb, tc
  WHERE ai_filter('{knob} unary ok? ' || x)
    AND ai_filter('{knob} edgeAB ok? ' || x || ' ' || y)
    AND ai_filter('{knob} edgeBC ok? ' || y || ' ' || z)
  ORDER BY x, y, z
) TO '{out}' (FORMAT CSV, HEADER);
COPY (SELECT (sum(llm_calls)-sum(embed_calls))::BIGINT AS chat FROM ai_usage())
  TO '{usage}' (FORMAT CSV, HEADER);
"""
    t0 = time.time()
    r = subprocess.run([BIN, "-batch"], input=sql, capture_output=True, text=True, timeout=1800)
    wall = time.time() - t0
    if r.returncode != 0:
        print(r.stderr[-1500:])
        sys.exit(f"{tag}: duckdb failed")
    rows = open(out).read().splitlines()[1:]
    chat = int(float(open(usage).read().splitlines()[1]))
    for f in (out, usage):
        os.remove(f)
    return rows, chat, wall


def sweep(points):
    print(f"{'N':>4} {'pass%':>6} {'mode':<11} {'chat':>6} {'wall':>8} {'rows':>6} {'calls_x':>8} {'wall_x':>7}")
    for n, p in points:
        base = None
        for mode, settings in MODES.items():
            rows, chat, wall = run_mode(settings, n, p, f"{mode.replace('+','_')}_{n}_{p}")
            if base is None:
                base = (rows, chat, wall)
                print(f"{n:>4} {p:>6} {mode:<11} {chat:>6} {wall:>7.1f}s {len(rows):>6}")
            else:
                assert rows == base[0], f"N={n} p={p} {mode}: results differ from set mode!"
                print(f"{n:>4} {p:>6} {mode:<11} {chat:>6} {wall:>7.1f}s {len(rows):>6} "
                      f"{chat / base[1]:>7.2f}x {wall / base[2]:>6.2f}x")


def star(weak_side):
    """Star topology (A center, edges A-B and A-C); one partner side is lethal (value-embedded
    [p=10]). Fixed schedules must guess which edge to run first; adaptive learns it -- the
    discriminating case for per-member reordering (identical results asserted)."""
    marker_b = "||'[p=10]'" if weak_side == "b" else ""
    marker_c = "||'[p=10]'" if weak_side == "c" else ""
    print(f"star weak={weak_side}:")
    base = None
    for mode, settings in MODES.items():
        if mode.startswith("tuple"):
            continue
        sql = f"""
CALL ai_mock_start();
{settings}
CREATE TABLE ta AS SELECT 'a'||i::VARCHAR AS x FROM range(12) r(i);
CREATE TABLE tb AS SELECT 'b'||i::VARCHAR{marker_b} AS y FROM range(12) r(i);
CREATE TABLE tc AS SELECT 'c'||i::VARCHAR{marker_c} AS z FROM range(12) r(i);
SELECT ai_usage_reset();
COPY (SELECT * FROM ta, tb, tc
  WHERE ai_filter('[p=70][sleep={SLEEP}] ab? '||x||' '||y)
    AND ai_filter('[p=70][sleep={SLEEP}] ac? '||x||' '||z) ORDER BY x, y, z)
  TO '/tmp/svt_star.csv' (FORMAT CSV, HEADER);
COPY (SELECT (sum(llm_calls)-sum(embed_calls))::BIGINT AS chat FROM ai_usage())
  TO '/tmp/svt_star_u.csv' (FORMAT CSV, HEADER);
"""
        t0 = time.time()
        r = subprocess.run([BIN, "-batch"], input=sql, capture_output=True, text=True, timeout=1800)
        wall = time.time() - t0
        if r.returncode != 0:
            print(r.stderr[-800:])
            sys.exit(f"star {mode}: failed")
        rows = open('/tmp/svt_star.csv').read().splitlines()[1:]
        chat = int(float(open('/tmp/svt_star_u.csv').read().splitlines()[1]))
        if base is None:
            base = rows
        else:
            assert rows == base, f"star weak={weak_side} {mode}: results differ!"
        print(f"  {mode:<13} {chat:>6} calls  {wall:>5.1f}s  {len(rows):>5} rows")


if __name__ == "__main__":
    # size sweep at the historic 50% pass rate, then a selectivity sweep at fixed size
    sweep([(8, 50), (16, 50), (24, 50)])
    sweep([(16, 20), (16, 80)])
    star("b")
    star("c")
    print("ALL MODES RESULT-IDENTICAL")
