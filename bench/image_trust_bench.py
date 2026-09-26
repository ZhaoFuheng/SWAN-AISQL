#!/usr/bin/env python3
"""Test the all-image always-evaluate safety rule on a tiny 4-way image join (q11 shape).

Claim under test: a speculative node whose rows' unknown leaves are ALL images must always
evaluate them (pruning the join input); trusting a fictitious estimate instead passes every
row through unpruned and the join's pair domain explodes.

Modes (same query, same data):
  guarded  -- default: all-image rows always evaluate at the leaf (prune before the join)
  no_spec  -- threshold=0 bypasses EMITTING the leaf nodes at plan time (pure pull-up): the
              un-pruned upper bound; the top node sees the full cross product's pair domain
  trusting -- ai_debug_trust_image_estimate=1, default threshold 0.5, MLP pre-warmed on a text
              query: leaf nodes exist and the gate consults the estimate for all-image rows.
              The mock's labels are uncorrelated with its embed vectors, so the warm estimate
              is uninformative (~base rate) -- trusting it passes rows through un-pruned and
              converges toward the no_spec explosion.

Asserts: identical result rows in all modes (the pulled-up node re-checks everything);
no_spec >> guarded calls; trusting reported (between guarded and no_spec).
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("DUCKDB_BIN", os.path.abspath(os.path.join(HERE, "../build/release/duckdb")))
N = int(os.environ.get("IMG_TRUST_N", "10"))  # rows per table

def make_images(root, n=None):
    n = n or N
    paths = {}
    for t in ["a", "b", "c", "d"]:
        paths[t] = []
        for i in range(n):
            p = os.path.join(root, f"{t}{i}.png")
            with open(p, "wb") as f:
                f.write(hashlib.sha256(f"{t}{i}".encode()).digest() * 2)  # 64 deterministic bytes
            paths[t].append(p)
    return paths

def run_mode(imgroot, paths, settings, tag, warmup=""):
    values = {t: ", ".join(f"({i}, '{paths[t][i]}')" for i in range(len(paths[t]))) for t in paths}
    out = os.path.join(imgroot, f"out_{tag}.csv")
    usage = os.path.join(imgroot, f"usage_{tag}.csv")
    sql = f"""
CALL ai_mock_start();
{settings}
CREATE TABLE ta(id INT, img VARCHAR); INSERT INTO ta VALUES {values['a']};
CREATE TABLE tb(id INT, img VARCHAR); INSERT INTO tb VALUES {values['b']};
CREATE TABLE tc(id INT, img VARCHAR); INSERT INTO tc VALUES {values['c']};
CREATE TABLE td(id INT, img VARCHAR); INSERT INTO td VALUES {values['d']};
{warmup}
SELECT ai_usage_reset();
COPY (
  WITH fa AS (SELECT * FROM ta WHERE ai_filter('leaf A ok? ' || ai_image(img))),
       fb AS (SELECT * FROM tb WHERE ai_filter('leaf B ok? ' || ai_image(img))),
       fc AS (SELECT * FROM tc WHERE ai_filter('leaf C ok? ' || ai_image(img))),
       fd AS (SELECT * FROM td WHERE ai_filter('leaf D ok? ' || ai_image(img)))
  SELECT fa.id AS a, fb.id AS b, fc.id AS c, fd.id AS d
  FROM fa, fb, fc, fd
  WHERE ai_filter('pair1 same? ' || ai_image(fa.img) || ' vs ' || ai_image(fb.img))
    AND ai_filter('pair2 same? ' || ai_image(fb.img) || ' vs ' || ai_image(fc.img))
    AND ai_filter('pair3 same? ' || ai_image(fc.img) || ' vs ' || ai_image(fd.img))
  ORDER BY a, b, c, d
) TO '{out}' (FORMAT CSV, HEADER);
COPY (SELECT sum(llm_calls) - sum(embed_calls) AS chat, sum(embed_calls) AS embed,
             sum(cache_hits) AS hits FROM ai_usage())
  TO '{usage}' (FORMAT CSV, HEADER);
"""
    t0 = time.time()
    r = subprocess.run([BIN, "-batch"], input=sql, capture_output=True, text=True, timeout=600)
    wall = time.time() - t0
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit(f"{tag}: duckdb failed")
    rows = open(out).read().splitlines()[1:] if os.path.exists(out) else []
    vals = open(usage).read().splitlines()[1].split(",")
    return rows, int(float(vals[0])), int(float(vals[1])), wall, int(float(vals[2]))

def scale():
    """Sweep table size for guarded vs no_spec only: how the un-pruned gap grows."""
    print(f"{'N/table':>8} {'mode':<8} {'chat':>6} {'cache_hits':>10} {'wall':>7} {'rows':>6} "
          f"{'call_ratio':>10} {'wall_ratio':>10}")
    OFF = "SET ai_join_factorize='off';"
    for n in [6, 10, 14, 18, 22, 26]:
        with tempfile.TemporaryDirectory() as root:
            paths = make_images(root, n)
            g = run_mode(root, paths, OFF, f"g{n}")
            u = run_mode(root, paths, OFF + "SET ai_debug_speculative_threshold=0;", f"u{n}")
            f = run_mode(root, paths, "", f"f{n}")
        assert g[0] == u[0] and g[0] == f[0], f"N={n}: results differ!"
        print(f"{n:>8} {'guarded':<8} {g[1]:>6} {g[4]:>10} {g[3]:>6.1f}s {len(g[0]):>6}")
        print(f"{n:>8} {'no_spec':<8} {u[1]:>6} {u[4]:>10} {u[3]:>6.1f}s {len(u[0]):>6} "
              f"{u[1] / g[1]:>9.1f}x {u[3] / g[3]:>9.1f}x")
        print(f"{n:>8} {'graph':<8} {f[1]:>6} {f[4]:>10} {f[3]:>6.1f}s {len(f[0]):>6} "
              f"{f[1] / g[1]:>9.1f}x {f[3] / g[3]:>9.1f}x")
    sys.exit(0)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "scale":
        scale()
    with tempfile.TemporaryDirectory() as root:
        paths = make_images(root)
        # The first four modes pin ai_join_factorize='off' so they exercise the REGION path
        # (fold the expanded cross product) -- the regime where the always-evaluate safety rule
        # matters. The factor_graph mode runs the defaults: the n-ary factor graph prunes
        # exactly, so the speculative distinction disappears entirely.
        OFF = "SET ai_join_factorize='off';"
        guarded = run_mode(root, paths, OFF, "guarded")
        no_spec = run_mode(root, paths, OFF + "SET ai_debug_speculative_threshold=0;", "no_spec")
        # Warm the MLP on a text query first so the gate is live (not in cold warm-up) when the
        # all-image rows arrive -- the regime the safety rule protects.
        warm = ("CREATE TABLE wtxt AS SELECT 'doc '||range::VARCHAR AS d FROM range(200);\n"
                "CREATE TABLE _w AS SELECT d FROM wtxt WHERE ai_filter('is '||d||' fine?')"
                " AND ai_filter('is '||d||' good?');\n"
                # burn ~a second so the async trainer fires a step before the image query
                # (the warm latch reads train_steps; racing it leaves the gate in cold warm-up)
                "CREATE TABLE _delay AS SELECT count(*) FROM range(200000000);\n")
        trusting = run_mode(root, paths, OFF + "SET ai_debug_trust_image_estimate=true;", "trusting",
                            warmup=warm)
        # Same, with a threshold below the arbitrary estimate: the gate now PASSES all-image
        # rows on that estimate -- the exact failure mode the safety rule prevents.
        trusting_low = run_mode(
            root, paths,
            OFF + "SET ai_debug_trust_image_estimate=true; SET ai_debug_speculative_threshold=0.1;",
            "trusting_low", warmup=warm)
        factor_graph = run_mode(root, paths, "", "factor_graph")

    print(f"{'mode':<12} {'chat':>6} {'cache_hits':>10} {'embed':>6} {'wall':>7} {'rows':>5}")
    for tag, (rows, chat, embed, wall, hits) in [("guarded", guarded), ("no_spec", no_spec),
                                                 ("trusting", trusting), ("trusting_low", trusting_low),
                                                 ("factor_graph", factor_graph)]:
        print(f"{tag:<12} {chat:>6} {hits:>10} {embed:>6} {wall:>6.1f}s {len(rows):>5}")

    ok = True
    if (guarded[0] != no_spec[0] or guarded[0] != trusting[0] or guarded[0] != trusting_low[0] or
            guarded[0] != factor_graph[0]):
        print("FAIL: result rows differ between modes (must be result-preserving)")
        ok = False
    if no_spec[1] <= int(guarded[1] * 1.5):
        print(f"FAIL: un-pruned baseline did not explode calls ({no_spec[1]} vs {guarded[1]})")
        ok = False
    # The trusting modes race the async trainer: a still-cold MLP forces evaluation (guarded-like),
    # a warm one passes rows on the arbitrary estimate (explodes). At least one variant must land
    # in the warm regime and show the failure mode.
    if max(trusting[1], trusting_low[1]) <= int(guarded[1] * 1.5):
        print(f"FAIL: neither trusting variant exploded ({trusting[1]}, {trusting_low[1]} vs {guarded[1]})")
        ok = False
    if factor_graph[1] > guarded[1]:
        print(f"FAIL: factor graph made more calls than guarded ({factor_graph[1]} vs {guarded[1]})")
        ok = False
    if ok:
        print(f"FACTOR GRAPH: {factor_graph[1]} calls / {factor_graph[3]:.1f}s -- exact pruning, "
              f"no speculative estimate needed")
        print(f"CLAIM HOLDS: un-pruned pair domain costs {no_spec[1]}/{guarded[1]} = "
              f"{no_spec[1] / guarded[1]:.1f}x chat calls, {no_spec[3] / guarded[3]:.0f}x wall; "
              f"trusting: {trusting[1]} calls / {trusting[3]:.1f}s; "
              f"trusting_low (gate passes): {trusting_low[1]} calls / {trusting_low[3]:.1f}s")
    sys.exit(0 if ok else 1)

main()
