#!/usr/bin/env python3
"""Per-query table of every recorded system (SWAN, PLOP, BlendSQL, Palimpzest, Sema, LOTUS) for one SemBench suite.

  python3 compare.py MOVIE            (from aisql-bench/; also ECOMM, MMQA)
  SWAN_TAG=_x python3 compare.py MOVIE   compare a tagged SWAN run (swan_movie_results_x.json)

Reads SUITE/<system>_<suite>_results.json (SWAN's with the tag) and prints a markdown table (quality, LLM calls,
latency, cost per query and in total) and the macro quality. The table compares the systems on one model with one
number of requests in flight, both read from the summaries: it is not printed when the runs differ. Under the
table it names every run that was not a replay at the recorded pace (timed live, replayed without the recorded
latency, or not recording which) and every query a system could not run. The Sema run must record Sema's shipped
settings and, for every query, the text of sema_queries/ it ran (sema_common.py).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sema_common  # noqa: E402 -- the settings a Sema run records, a query text's hash

QUALITY_KEYS = ("quality", "f1", "F1", "ari")


def quality(rec):
    return next((rec[k] for k in QUALITY_KEYS if k in rec and rec[k] is not None), None)


def fmt(v, digits):
    return ("%.*f" % (digits, v)) if isinstance(v, (int, float)) else "-"


def in_flight(summary):
    """The requests a run kept in flight, as its runner recorded them."""
    return next((v for v in (summary.get("concurrency"), summary.get("workers")) if v is not None), None)


def run_modes(per_query):
    """How a run's latencies were measured, one word per kind of record: the runner's own `timing` where it wrote
    one, else "replay" when no call was new to the cache and "live" when some were, and "timing not recorded" for
    a record that says neither."""
    modes = set()
    for r in per_query.values():
        fresh = r.get("llm_calls_fresh")
        modes.add(r.get("timing") or ("timing not recorded" if fresh is None else "live" if fresh else "replay"))
    return modes or {"timing not recorded"}


def sema_problem(suite, summary):
    """Why a Sema summary cannot stand in the table, or None: it must record Sema's shipped default settings (the
    runner checks the binary against them before it runs) and, per query, the text it ran."""
    settings = summary.get("settings")
    if not isinstance(settings, dict) or set(settings) != set(sema_common.SETTINGS):
        return "it does not record Sema's settings as the runner does (an earlier version of the runner wrote it)"
    if settings != sema_common.SETTINGS:
        other = {k: v for k, v in settings.items() if v != sema_common.SETTINGS[k]}
        return f"it records settings other than Sema's shipped defaults: {other}"
    stale = sorted(q for q, r in summary["per_query"].items()
                   if r.get("query_sha256") != sema_common.sha256_of(os.path.join(HERE, suite, "sema_queries", q + ".sql")))
    return f"{', '.join(stale)} ran another text than {suite}/sema_queries/ now holds" if stale else None


def shared(summaries, what, value):
    """The value every run records. The table compares the systems on one model with one number of requests in
    flight, so it is not printed when the runs differ or one does not say."""
    got = {name: value(s) for name, s in summaries}
    got = {name: None if v is None else str(v) for name, v in got.items()}
    if None in got.values() or len(set(got.values())) != 1:
        sys.exit(f"the recorded runs do not share one {what}, so they are not one comparison: "
                 + ", ".join(f"{name} {v or 'not recorded'}" for name, v in got.items()))
    return next(iter(got.values()))


SYSTEMS = (("LOTUS", "lotus"), ("BlendSQL", "blendsql"), ("Palimpzest", "palimpzest"), ("PLOP", "plop"), ("Sema", "sema"),
           ("SWAN", "swan"))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    suite = sys.argv[1].upper()
    low = suite.lower()
    tag = os.environ.get("SWAN_TAG", "")
    summaries = []
    for name, sysname in SYSTEMS:
        path = os.path.join(HERE, suite, f"{sysname}_{low}_results{tag if sysname == 'swan' else ''}.json")
        if os.path.exists(path):
            summaries.append((name, json.load(open(path))))
    if not summaries:
        sys.exit(f"no results for {suite} under {os.path.join(HERE, suite)}")
    for name, summary in summaries:
        problem = sema_problem(suite, summary) if name == "Sema" else None
        if problem:
            sys.exit(f"{suite}/sema_{low}_results.json cannot stand in the table: {problem}; it shows Sema at its "
                     "default settings only, as sema_movie.py records a run")
    model = shared(summaries, "model", lambda s: s.get("model"))
    flight = shared(summaries, "number of requests in flight", in_flight)
    runs = [(name, s["per_query"]) for name, s in summaries]
    # every query any system ran, in the first system's order; a system that did not run one shows n/a
    # (PLOP and BlendSQL do not support images: ECOMM / MMQA are not run for them)
    queries = list(runs[0][1])
    for _, r in runs[1:]:
        queries += [q for q in r if q not in queries]
    counted = {n: 0 for n, _ in runs}
    names = [n for n, _ in runs]
    print(f"# SemBench {suite}: " + " vs ".join(names) + f" ({model}, {flight} requests in flight"
          + (f"; SWAN {tag}" if tag else "") + ")\n")
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
        cells = ["n/a" if not rec else "failed" if rec.get("error") else fmt(quality(rec), 3) for _, rec in recs]
        cells += [f"{rec.get('llm_calls') or 0:,}" if rec else "n/a" for _, rec in recs]
        cells += [fmt(rec.get("latency_s"), 1) if rec else "n/a" for _, rec in recs]
        cells += [fmt(rec.get("cost_usd"), 3) if rec else "n/a" for _, rec in recs]
        print(f"| {q} | " + " | ".join(cells) + " |")

    def agg(x, text):  # an aggregate over fewer queries than the suite says how many it covers
        return text if counted[x] == len(queries) else f"{text} ({counted[x]}q)"
    cells = [agg(x, f"**{tot[x][0] / counted[x]:.3f}**") if counted[x] else "n/a" for x in names] \
        + [agg(x, f"{tot[x][1]:,}") for x in names] + [agg(x, f"{tot[x][2]:.0f}") for x in names] \
        + [agg(x, f"{tot[x][3]:.2f}") for x in names]
    print("| **macro / Σ** | " + " | ".join(cells) + " |")
    if any(counted[x] != len(queries) for x in names):
        print("\nA macro or sum over fewer queries than the suite (marked with its query count) is not comparable with "
              "the full-suite ones.")
    marked = [f"{n} ({'/'.join(sorted(run_modes(r)))})" for n, r in runs if run_modes(r) != {"replay"}]
    if marked:
        print("\nRuns whose records do not show a replay at the recorded pace: " + ", ".join(marked) + ".")
    failed = [f"{n} {q}" for q in queries for n, r in runs if (r.get(q) or {}).get("error")]
    if failed:
        print("\nfailed marks a query a system could not run (it scores 0; what it had spent is in the sums): "
              + ", ".join(failed) + ".")
    if any(n in ("PLOP", "BlendSQL", "Palimpzest") for n in names):
        print("\nPLOP, BlendSQL and Palimpzest costs are estimates from their token counts (their requests carry no provider cost header).")


if __name__ == "__main__":
    main()
