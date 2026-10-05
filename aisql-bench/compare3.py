#!/usr/bin/env python3
"""Three-way per-query table: SWAN (default), SWAN+Jev (_typesafe tag), LOTUS.  python3 compare3.py SUITE"""
import json, sys
suite = sys.argv[1]; l = suite.lower()
def load(p): return json.load(open(f"{suite}/{p}"))["per_query"]
s, j, lo = load(f"swan_{l}_results.json"), load(f"swan_{l}_results_typesafe.json"), load(f"lotus_{l}_results.json")
Q = ("quality", "f1", "F1", "ari")
q = lambda x: next((x[c] for c in Q if c in x and x[c] is not None), None)
f = lambda v, p: ("%.*f" % (p, v)) if isinstance(v, (int, float)) else "-"
print("| q | SWAN | SWAN+Jev | LOTUS | SWAN calls | +Jev calls | LOTUS calls | SWAN lat (s) | +Jev lat (s) | LOTUS lat (s) | SWAN $ | +Jev $ | LOTUS $ |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
tot = {n: [0, 0, 0, 0] for n in "sjl"}
for qq in s:
    cells = []
    for n, d in (("s", s), ("j", j), ("l", lo)):
        x = d.get(qq, {}); v = (q(x), x.get("llm_calls"), x.get("latency_s"), x.get("cost_usd"))
        for i in range(4): tot[n][i] += v[i] or 0
        cells.append(v)
    print(f"| {qq} | {f(cells[0][0],3)} | {f(cells[1][0],3)} | {f(cells[2][0],3)} | {cells[0][1]:,} | {cells[1][1]:,} | {cells[2][1]:,} | "
          f"{f(cells[0][2],1)} | {f(cells[1][2],1)} | {f(cells[2][2],1)} | {f(cells[0][3],4)} | {f(cells[1][3],4)} | {f(cells[2][3],4)} |")
n = len(s)
print(f"| **macro / Σ** | **{tot['s'][0]/n:.3f}** | **{tot['j'][0]/n:.3f}** | **{tot['l'][0]/n:.3f}** | {tot['s'][1]:,} | {tot['j'][1]:,} | {tot['l'][1]:,} | "
      f"{tot['s'][2]:,.0f} | {tot['j'][2]:,.0f} | {tot['l'][2]:,.0f} | {tot['s'][3]:.2f} | {tot['j'][3]:.2f} | {tot['l'][3]:.2f} |")
