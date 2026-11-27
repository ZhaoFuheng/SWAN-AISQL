#!/usr/bin/env python3
"""Q1-Q30 table of every system: LOTUS, BlendSQL, Palimpzest, Sema, PLOP-DP and SWAN, scored by the deterministic
scorer (eval_agentbench.py: LIMIT-free PLOP ground truth, LIMIT queries sound+complete).

Reads results/{<SWAN_TAG>,plop,lotus,blendsql,palimpzest,<SEMA_TAG>}_agentbench_results.json for calls / latency /
cost and writes results/agentbench_comparison_three_way.md.

  SWAN_TAG=swan_20261111 python3 report_three_way.py     (the committed table; SWAN_TAG=swan for your own run)
  SEMA_TAG picks the Sema run the same way (default sema, the committed one); it must be Sema at its defaults.

The table compares the systems on one model with one number of requests in flight, both read from the summaries:
it is not written when the selected runs (or the ground truth's model) differ. A query a system did not run is n/a
(eval_agentbench.py decides, so the two commands agree) and its aggregates say how many queries they cover; a query
a system could not run is "failed" (it scores 0). A run whose records show that it was timed live (some of its
calls were new to the cache, llm_calls_fresh) or replayed without the recorded latency is marked. The Sema run
must record Sema's shipped settings and, for every query, the text of sema_queries/ it ran (sema_common.py).
"""
import importlib.util
import json
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sema_common  # noqa: E402 -- aisql-bench/sema_common.py: the settings a Sema run records, a query text's hash
SWAN_TAG = os.environ.get("SWAN_TAG", "swan")
SEMA_TAG = os.environ.get("SEMA_TAG", "sema")


def summary_path(tag):
    return os.path.join(HERE, "results", tag + "_agentbench_results.json")


# LOTUS, PLOP and SWAN are always shown; BlendSQL, Palimpzest and Sema when they have a run. A run that was asked
# for by name must exist: a mistyped tag stops the report instead of dropping the column.
SYSTEMS = [("LOTUS", "lotus")]
for name, tag in (("BlendSQL", "blendsql"), ("Palimpzest", "palimpzest"), ("Sema", SEMA_TAG)):
    if os.path.exists(summary_path(tag)) or (name == "Sema" and "SEMA_TAG" in os.environ):
        SYSTEMS.append((name, tag))
SYSTEMS += [("PLOP", "plop"), ("SWAN", SWAN_TAG)]
for name, tag in SYSTEMS:
    if not os.path.exists(summary_path(tag)):
        raise SystemExit(f"no {name} run at results/{tag}_agentbench_results.json"
                         + (" (set SWAN_TAG to a recorded run; the committed one is swan_20261111)" if name == "SWAN" else ""))

ev_src = open(os.path.join(HERE, "eval_agentbench.py")).read().replace("\nmain()\n", "\n")
ev = importlib.util.module_from_spec(importlib.util.spec_from_loader("ev", loader=None))
ev.__dict__["__file__"] = os.path.join(HERE, "eval_agentbench.py")
exec(compile(ev_src, "eval_agentbench.py", "exec"), ev.__dict__)

HEADER = """# agent_bench Q1-Q30: {title} ({model}) -- {today}

Quality = deterministic scorer (eval_agentbench.py): LIMIT-free PLOP execution as ground truth; LIMIT queries scored sound+complete. Calls = true request counts through the cache proxy. Each system's columns are {coverage} of the 30 queries ({runs}) with {in_flight} requests in flight, {timing}, one system at a time; SWAN replays PLOP's samples via the shared-verdict alias. LOTUS translations are ours (aisql-bench/README.md, provenance table). SWAN run: results/{swan_tag}.{sema_run}

**Read the LOTUS quality column with care.** SWAN and PLOP consume the *same* recorded verdict for every
shared prompt (the proxy's shared-verdict alias), so they agree with the ground truth by construction
wherever their plans agree. LOTUS uses its own prompt templates, so its verdicts are independent samples of
the same model: on queries whose answer hinges on a handful of judgments ({empty} have an EMPTY
ground truth; LOTUS admitted {lotus_rows} rows) any disagreement scores 0 even when LOTUS's answer
is a perfectly reasonable reading. Its macro therefore measures verdict agreement with PLOP's samples as
much as correctness; the calls / latency / cost columns are the like-for-like comparison. BlendSQL (our
mechanical translation, translate_to_blendsql.py; zero-shot; cost estimated from its token counts) is in
the same position as LOTUS: independent samples. Palimpzest (Abacus optimizer; the LOTUS
programs with their semantic operators served by Palimpzest; cost estimated from its token counts) likewise{sema}.

"""
SEMA_CLAUSE = ",\nand Sema (the authors' fork at its default settings; PLOP's queries in its s'...' dialect, translate_to_sema.py)"
PARTIAL_NOTE = ("n/a marks a query a system did not run (or one without ground truth). A macro or sum over fewer than 30 "
                "queries is marked with the number it covers and is not comparable with the full-suite ones.")
FAILED_NOTE = ("failed marks a query a system could not run: it scores 0, and the calls, time and cost it had spent "
               "until then are in the sums.")


def table_header():
    names = [n for n, _ in SYSTEMS]
    cols = names + [f"{n} calls" for n in names] + [f"{n} lat (s)" for n in names] + [f"{n} $" for n in names]
    return "| q | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)


def run_days(per_query):
    days = sorted({(r.get("ran_at") or "")[:10] for r in per_query.values()} - {""})
    return "undated" if not days else days[0] if len(days) == 1 else f"{days[0]} to {days[-1]}"


def run_modes(per_query):
    """How a run's latencies were measured, one word per kind of record: the runner's own `timing` where it wrote
    one, else "live" when some of the query's calls were new to the cache (llm_calls_fresh) and "replay" when none
    were. A record without that count (written before its runner kept one, or with the proxy's counter out of
    reach) shows nothing to mark and reads as a replay, which is how every committed run was made."""
    modes = set()
    for r in per_query.values():
        modes.add(r.get("timing") or ("live" if r.get("llm_calls_fresh") else "replay"))
    return modes or {"replay"}


def n_run(per_query):
    return sum(1 for i in range(1, 31) if f"Q{i}" in per_query)


def n_failed(per_query):
    return sum(1 for i in range(1, 31) if (per_query.get(f"Q{i}") or {}).get("error"))


def run_label(name, run):
    parts = []
    if n_run(run["per_query"]) < 30:
        parts.append(f"{n_run(run['per_query'])} of 30 queries")
    if run_modes(run["per_query"]) != {"replay"}:
        parts.append("/".join(sorted(run_modes(run["per_query"]))))
    if n_failed(run["per_query"]):
        parts.append(f"{n_failed(run['per_query'])} failed")
    return f"{name} {run_days(run['per_query'])}" + (f" ({'; '.join(parts)})" if parts else "")


def sema_problem(summary):
    """Why a Sema summary cannot stand in the comparison, or None. It must record Sema's shipped default settings
    (the runner checks the binary against them before it runs) and, per query, the text it ran: rows of an earlier
    runner, of other settings or of an earlier query text are not Sema as the table describes it."""
    settings = summary.get("settings")
    if not isinstance(settings, dict) or set(settings) != set(sema_common.SETTINGS):
        return "it does not record Sema's settings as the runner does (an earlier version of the runner wrote it)"
    if settings != sema_common.SETTINGS:
        other = {k: v for k, v in settings.items() if v != sema_common.SETTINGS[k]}
        return f"it records settings other than Sema's shipped defaults: {other}"
    stale = [q for q, r in summary["per_query"].items()
             if r.get("query_sha256") != sema_common.sha256_of(os.path.join(HERE, "sema_queries", q + ".sql"))]
    if stale:
        return f"{', '.join(sorted(stale, key=lambda q: int(q[1:])))} ran another text than sema_queries/ now holds"
    return None


def shared(summaries, what, keys):
    """The value every summary records under one of `keys`. The table compares the systems on one model with one
    number of requests in flight, so it is not written when the selected runs differ or one does not say."""
    got = {name: next((str(s[k]) for k in keys if s.get(k) is not None), None) for name, s in summaries.items()}
    if None in got.values() or len(set(got.values())) != 1:
        raise SystemExit(f"the selected runs do not share one {what}, so they are not one comparison: "
                         + ", ".join(f"{name} {value or 'not recorded'}" for name, value in got.items()))
    return next(iter(got.values()))


def main():
    files = {name: json.load(open(summary_path(tag))) for name, tag in SYSTEMS}
    runs = {name: f["per_query"] for name, f in files.items()}
    gt = ({f"the ground truth ({ev.GT_TAG})": json.load(open(summary_path(ev.GT_TAG)))}
          if os.path.exists(summary_path(ev.GT_TAG)) else {})
    model = shared({**files, **gt}, "model", ("model",))
    in_flight = shared(files, "number of requests in flight", ("concurrency", "workers"))
    # the header's facts are read from the result files, so a regenerated table never cites an older run
    empty = [f"Q{i}" for i in range(1, 31) if not sum(ev.scorer.load_rows(ev.GT_TAG, f"Q{i}")[1].values())]
    lotus_rows = " / ".join(f"{sum(ev.scorer.load_rows('lotus', q)[1].values()):,}" for q in empty)
    title = " vs ".join("PLOP-DP" if n == "PLOP" else n for n, _ in SYSTEMS)
    covered = {n: n_run(runs[n]) for n, _ in SYSTEMS}
    partial = [n for n, _ in SYSTEMS if covered[n] < 30]
    unusual = [n for n, _ in SYSTEMS if run_modes(runs[n]) != {"replay"}]
    timing = "a replay of its recorded calls with their recorded response times" + (
        f" (except as marked for {', '.join(unusual)})" if unusual else "")
    sema = sema_run = ""
    if "Sema" in files:
        problem = sema_problem(files["Sema"])
        if problem:
            raise SystemExit(f"results/{SEMA_TAG} cannot stand in the comparison: {problem}; the table shows Sema at "
                             "its default settings only, as sema_agentbench.py records a run")
        sema, sema_run = SEMA_CLAUSE, f" Sema run: results/{SEMA_TAG}."
    lines = [HEADER.format(title=title, model=model, in_flight=in_flight, today=date.today().isoformat(),
                           swan_tag=SWAN_TAG, sema_run=sema_run,
                           coverage="one run" if partial else "one full run", timing=timing, sema=sema,
                           runs=", ".join(run_label(name, files[name]) for name, _ in SYSTEMS),
                           empty="/".join(empty), lotus_rows=lotus_rows).rstrip("\n"), "", table_header()]
    q_sum = {n: [] for n, _ in SYSTEMS}
    tot = {n: {"calls": 0, "lat": 0.0, "cost": 0.0} for n, _ in SYSTEMS}
    for i in range(1, 31):
        q = f"Q{i}"
        cells = []
        for name, tag in SYSTEMS:
            v, _ = ev.score(tag, q)
            if v is None:  # not run (or no ground truth): no observation, not a zero
                cells.append("n/a")
                continue
            q_sum[name].append(v)
            cells.append("failed" if ev.scorer.failed(tag, q) else f"{v:.3f}")
        for key, fmt in (("calls", "{:,}"), ("lat", "{:.1f}"), ("cost", "{:.4f}")):
            for name, _ in SYSTEMS:
                r = runs[name].get(q)
                if r is None:
                    cells.append("n/a")
                    continue
                val = {"calls": r["llm_calls"], "lat": r["latency_s"],
                       "cost": r.get("cost_usd", r.get("cost_usd_est", 0.0))}[key]
                tot[name][key] += val
                cells.append(fmt.format(val))
        lines.append(f"| {q} | " + " | ".join(cells) + " |")

    failed = {n: n_failed(runs[n]) for n, _ in SYSTEMS}

    def agg(text, k, n=None):  # an aggregate says how many queries it covers when not 30, and how many of them failed
        marks = ([f"{k}q"] if k != 30 else []) + ([f"{failed[n]} failed"] if n and failed[n] else [])
        return f"{text} ({'; '.join(marks)})" if marks else text
    macro = " | ".join(agg(f"**{sum(q_sum[n]) / len(q_sum[n]):.3f}**", len(q_sum[n])) if q_sum[n] else "n/a"
                       for n, _ in SYSTEMS)
    calls = " | ".join(agg(f"{tot[n]['calls']:,}", covered[n], n) for n, _ in SYSTEMS)
    lat = " | ".join(agg(f"{tot[n]['lat']:.0f}", covered[n], n) for n, _ in SYSTEMS)
    cost = " | ".join(agg(f"{tot[n]['cost']:.3f}", covered[n], n) for n, _ in SYSTEMS)
    lines.append(f"| **macro / Σ** | {macro} | {calls} | {lat} | {cost} |")
    if partial or any(len(q_sum[n]) < 30 for n, _ in SYSTEMS):
        lines += ["", PARTIAL_NOTE]
    if any(failed.values()):
        lines += ["", FAILED_NOTE]
    out = "\n".join(lines) + "\n"
    path = os.path.join(HERE, "results", "agentbench_comparison_three_way.md")
    open(path, "w").write(out)
    print(out)
    print(f"written -> {path}")


main()
