#!/usr/bin/env python3
"""Per-query SWAN comparison between two result tags: python3 compare_tags.py [base_tag] [other_tag]
(default: '' = the default run vs '_typesafe'). Reads <SUITE>/swan_<suite>_results<tag>.json."""
import json, sys
BASE = sys.argv[1] if len(sys.argv) > 1 else ""
OTHER = sys.argv[2] if len(sys.argv) > 2 else "_typesafe"
SUITES = ["MMQA", "ECOMM", "MOVIE"]
Q_KEYS = ("f1", "F1", "quality", "ari")
def rows(suite, tag):
    try:
        d = json.load(open(f"{suite}/swan_{suite.lower()}_results{tag}.json"))
    except FileNotFoundError:
        return {}
    out = {}
    for q, x in d["per_query"].items():
        qual = next((x[k] for k in Q_KEYS if k in x and x[k] is not None), None)
        out[q] = dict(qual=qual, calls=x["llm_calls"], lat=x["latency_s"], cost=x["cost_usd"], fresh=x.get("llm_calls_fresh"))
    return out
def f(v, w, p=3): return ("%*.*f" % (w, p, v)) if isinstance(v, (int, float)) else "%*s" % (w, "-")
for suite in SUITES:
    a, b = rows(suite, BASE), rows(suite, OTHER)
    if not b:
        print(f"\n## {suite}: no '{OTHER}' results yet"); continue
    print(f"\n## {suite}  ('{BASE or 'default'}' -> '{OTHER}')")
    print("| q | quality | calls | latency (s) | cost ($) | fresh |"); print("|---|---|---|---|---|---|")
    sq = nq = 0; sc = sl = sd = 0.0
    for q in b:
        x, y = a.get(q, {}), b[q]
        dq = "" if x.get("qual") is None or y["qual"] is None else f" ({y['qual']-x['qual']:+.3f})"
        print(f"| {q} | {f(x.get('qual'),5)} -> {f(y['qual'],5)}{dq} | {x.get('calls','-')} -> {y['calls']} | {f(x.get('lat'),6,1)} -> {f(y['lat'],6,1)} | {f(x.get('cost'),6,4)} -> {f(y['cost'],6,4)} | {y.get('fresh','-')} |")
        if isinstance(y["qual"], (int, float)): sq += y["qual"]; nq += 1
        sc += y["calls"] or 0; sl += y["lat"] or 0; sd += y["cost"] or 0
    ma = [v["qual"] for v in a.values() if isinstance(v["qual"], (int, float))]
    print(f"| **macro / Σ** | {sum(ma)/len(ma) if ma else 0:.3f} -> {sq/nq if nq else 0:.3f} | {sum(v['calls'] or 0 for v in a.values())} -> {sc:.0f} | {sum(v['lat'] or 0 for v in a.values()):.0f} -> {sl:.0f} | {sum(v['cost'] or 0 for v in a.values()):.2f} -> {sd:.2f} | |")
