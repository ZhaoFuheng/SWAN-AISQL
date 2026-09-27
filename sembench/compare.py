#!/usr/bin/env python3
"""Per-query SWAN vs LOTUS table for one SemBench suite, from the recorded results files.

  python3 compare.py MOVIE            (from sembench/; also ECOMM, MMQA)
  SWAN_TAG=_x python3 compare.py MOVIE   compare a tagged SWAN run (swan_movie_results_x.json)

Reads SUITE/swan_<suite>_results[TAG].json and SUITE/lotus_<suite>_results.json, prints a markdown
table (quality, LLM calls, latency, cost per query and in total) and the macro quality.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QUALITY_KEYS = ("quality", "f1", "F1", "ari")


def load(path):
    return json.load(open(path))["per_query"]


def quality(rec):
    return next((rec[k] for k in QUALITY_KEYS if k in rec and rec[k] is not None), None)


def fmt(v, digits):
    return ("%.*f" % (digits, v)) if isinstance(v, (int, float)) else "-"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    suite = sys.argv[1].upper()
    low = suite.lower()
    tag = os.environ.get("SWAN_TAG", "")
    swan = load(os.path.join(HERE, suite, f"swan_{low}_results{tag}.json"))
    lotus = load(os.path.join(HERE, suite, f"lotus_{low}_results.json"))
    queries = [q for q in swan if q in lotus]
    print(f"# SemBench {suite}: SWAN{' (' + tag + ')' if tag else ''} vs LOTUS\n")
    print("| q | SWAN quality | LOTUS quality | SWAN calls | LOTUS calls | SWAN lat (s) | LOTUS lat (s) | SWAN $ | LOTUS $ |")
    print("|---|---|---|---|---|---|---|---|---|")
    tot = {"s": [0.0, 0, 0.0, 0.0], "l": [0.0, 0, 0.0, 0.0]}
    for q in queries:
        s, l = swan[q], lotus[q]
        for key, rec in (("s", s), ("l", l)):
            tot[key][0] += quality(rec) or 0.0
            tot[key][1] += rec.get("llm_calls") or 0
            tot[key][2] += rec.get("latency_s") or 0.0
            tot[key][3] += rec.get("cost_usd") or 0.0
        print(f"| {q} | {fmt(quality(s), 3)} | {fmt(quality(l), 3)} | {s.get('llm_calls', 0):,} | {l.get('llm_calls', 0):,} "
              f"| {fmt(s.get('latency_s'), 1)} | {fmt(l.get('latency_s'), 1)} | {fmt(s.get('cost_usd'), 3)} | {fmt(l.get('cost_usd'), 3)} |")
    n = len(queries)
    print(f"| **macro / Σ** | **{tot['s'][0] / n:.3f}** | **{tot['l'][0] / n:.3f}** | {tot['s'][1]:,} | {tot['l'][1]:,} "
          f"| {tot['s'][2]:.0f} | {tot['l'][2]:.0f} | {tot['s'][3]:.2f} | {tot['l'][3]:.2f} |")


if __name__ == "__main__":
    main()
