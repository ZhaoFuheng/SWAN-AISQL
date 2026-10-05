#!/usr/bin/env python3
"""Per-query table of every recorded system (SWAN, PLOP, BlendSQL, ThalamusDB, Palimpzest, LOTUS) for one SemBench suite.

  python3 compare.py MOVIE            (from aisql-bench/; also ECOMM, MMQA)
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


SYSTEMS = (("LOTUS", "lotus"), ("BlendSQL", "blendsql"), ("Palimpzest", "palimpzest"), ("ThalamusDB", "thalamusdb"), ("PLOP", "plop"), ("SWAN", "swan"))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    suite = sys.argv[1].upper()
    low = suite.lower()
    tag = os.environ.get("SWAN_TAG", "")
    runs = []
    for name, sysname in SYSTEMS:
        path = os.path.join(HERE, suite, f"{sysname}_{low}_results{tag if sysname == 'swan' else ''}.json")
        if os.path.exists(path):
            runs.append((name, load(path)))
    # every query any system ran, in the first system's order; a system that did not run one shows n/a
    # (PLOP and BlendSQL do not support images: ECOMM / MMQA are not run for them)
    queries = list(runs[0][1])
    for _, r in runs[1:]:
        queries += [q for q in r if q not in queries]
    counted = {n: 0 for n, _ in runs}
    names = [n for n, _ in runs]
    print(f"# SemBench {suite}: " + " vs ".join(names) + (f" (SWAN {tag})" if tag else "") + "\n")
    cols = [f"{n} quality" for n in names] + [f"{n} calls" for n in names] + [f"{n} lat (s)" for n in names] + [f"{n} $" for n in names]
    print("| q | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    tot = {n: [0.0, 0, 0.0, 0.0] for n in names}
    for q in queries:
        recs = [(n, r.get(q)) for n, r in runs]
        for n, rec in recs:
            if rec is None:
                continue
            counted[n] += 1
            tot[n][0] += quality(rec) or 0.0
            tot[n][1] += rec.get("llm_calls") or 0
            tot[n][2] += rec.get("latency_s") or 0.0
            tot[n][3] += rec.get("cost_usd") or 0.0
        cells = [fmt(quality(rec), 3) if rec else "n/a" for _, rec in recs]
        cells += [f"{rec.get('llm_calls') or 0:,}" if rec else "n/a" for _, rec in recs]
        cells += [fmt(rec.get("latency_s"), 1) if rec else "n/a" for _, rec in recs]
        cells += [fmt(rec.get("cost_usd"), 3) if rec else "n/a" for _, rec in recs]
        print(f"| {q} | " + " | ".join(cells) + " |")
    cells = [f"**{tot[x][0] / counted[x]:.3f}**" + ("" if counted[x] == len(queries) else f" ({counted[x]}q)") for x in names] \
        + [f"{tot[x][1]:,}" for x in names] + [f"{tot[x][2]:.0f}" for x in names] + [f"{tot[x][3]:.2f}" for x in names]
    print("| **macro / Σ** | " + " | ".join(cells) + " |")
    if any(counted[x] != len(queries) for x in names):
        print("\nA macro over fewer queries than the suite (marked with its query count) is not comparable with the "
              "full-suite macros.")
    if any(n in ("PLOP", "BlendSQL", "ThalamusDB", "Palimpzest") for n in names):
        print("\nPLOP, BlendSQL, ThalamusDB and Palimpzest costs are estimates from their token counts (their requests carry no provider cost header).")


if __name__ == "__main__":
    main()
