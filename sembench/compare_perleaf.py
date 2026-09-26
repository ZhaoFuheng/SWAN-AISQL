#!/usr/bin/env python3
"""Per-query SWAN comparison: re-record (2026-09-23, backup) vs per-leaf rerun (live result files)."""
import json, sys
SUITES = [("MMQA", "swan_mmqa_results.json"), ("ECOMM", "swan_ecomm_results.json"), ("MOVIE", "swan_movie_results.json")]
Q_KEYS = ("f1", "F1", "quality", "ari")
def rows(path):
    d = json.load(open(path))
    out = {}
    for q, x in d["per_query"].items():
        qual = next((x[k] for k in Q_KEYS if k in x and x[k] is not None), None)
        out[q] = dict(qual=qual, calls=x["llm_calls"], lat=x["latency_s"], cost=x["cost_usd"], fresh=x.get("llm_calls_fresh"))
    return out
def f(v, w): return ("%*.3f" % (w, v)) if isinstance(v, (int, float)) else "%*s" % (w, "-")
tot = []
for suite, fn in SUITES:
    a, b = rows(f"_rerecord_20260923_swan_results_backup/{fn}"), rows(f"{suite}/{fn}")
    print(f"\n## {suite}  (re-record 2026-09-23  ->  per-leaf rerun)")
    print("| q | quality | calls | latency (s) | cost ($) | fresh |"); print("|---|---|---|---|---|---|")
    sq = sc = sl = sd = 0; nq = 0
    for q in b:
        x, y = a.get(q, {}), b[q]
        dq = "" if x.get("qual") is None else f" ({y['qual']-x['qual']:+.3f})"
        dc = "" if x.get("calls") is None else f" ({y['calls']-x['calls']:+d})"
        dl = "" if x.get("lat") is None else f" ({y['lat']-x['lat']:+.0f})"
        print(f"| {q} | {f(x.get('qual'),5)} -> {f(y['qual'],5)}{dq} | {x.get('calls','-')} -> {y['calls']}{dc} | {f(x.get('lat'),6)} -> {f(y['lat'],6)}{dl} | {f(x.get('cost'),5)} -> {f(y['cost'],5)} | {y.get('fresh','-')} |")
        if isinstance(y["qual"], (int, float)): sq += y["qual"]; nq += 1
        sc += y["calls"] or 0; sl += y["lat"] or 0; sd += y["cost"] or 0
    ma = [v["qual"] for v in a.values() if isinstance(v["qual"], (int, float))]
    print(f"| **macro** | {sum(ma)/len(ma):.3f} -> {sq/nq:.3f} | {sum(v['calls'] or 0 for v in a.values())} -> {sc} | {sum(v['lat'] or 0 for v in a.values()):.0f} -> {sl:.0f} | {sum(v['cost'] or 0 for v in a.values()):.2f} -> {sd:.2f} | |")
