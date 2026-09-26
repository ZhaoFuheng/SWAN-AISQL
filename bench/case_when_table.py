#!/usr/bin/env python3
"""Render case_when_bench_results.json as a markdown table.

Delta = (on - off) / off * 100, i.e. the percent change going from reorder OFF to ON.
NEGATIVE = fewer LLM calls / less latency (improvement); POSITIVE = regression. Signs are always shown.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def delta(off, on):
    return 0.0 if off == 0 else 100.0 * (on - off) / off


def main():
    data = json.load(open(f"{HERE}/case_when_bench_results.json"))
    to_off = to_on = lo_off = lo_on = 0.0
    rows = []
    for e in data:
        off, on = e["off"], e["on"]
        co, cn, lo, ln = off["calls"], on["calls"], off["lat"], on["lat"]
        to_off += co
        to_on += cn
        lo_off += lo
        lo_on += ln
        rows.append((e["name"], e["leaves"], co, cn, delta(co, cn), lo, ln, delta(lo, ln), e["match"]))

    print("| query | leaves | LLM calls off→on | calls Δ | latency off→on | latency Δ | result |")
    print("|---|---|---|---|---|---|---|")
    for n, lv, co, cn, cd, lo, ln, ld, m in rows:
        print(f"| {n} | {lv} | {co} → {cn} | {cd:+.0f}% | {lo:.1f}s → {ln:.1f}s | {ld:+.0f}% | "
              f"{'✅ match' if m else '❌ CHANGED'} |")
    cd, ld = delta(to_off, to_on), delta(lo_off, lo_on)
    print(f"| **TOTAL** | | **{int(to_off)} → {int(to_on)}** | **{cd:+.0f}%** | "
          f"**{lo_off:.1f}s → {lo_on:.1f}s** | **{ld:+.0f}%** | ✅ |")
    print("\nΔ = (on − off)/off · 100 — negative = fewer calls / less latency (better), positive = regression.")


if __name__ == "__main__":
    main()
