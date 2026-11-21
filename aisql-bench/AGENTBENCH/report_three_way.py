#!/usr/bin/env python3
"""Q1-Q30 table of every system: LOTUS, BlendSQL, Palimpzest, PLOP-DP and SWAN, scored by the deterministic
scorer (eval_agentbench.py: LIMIT-free PLOP ground truth, LIMIT queries sound+complete).

Reads results/{<SWAN_TAG>,plop,lotus,blendsql,palimpzest}_agentbench_results.json for calls / latency / cost
and writes results/agentbench_comparison_three_way.md.

  SWAN_TAG=swan_20261111 python3 report_three_way.py     (the committed table; SWAN_TAG=swan for your own run)
"""
import importlib.util
import json
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
SWAN_TAG = os.environ.get("SWAN_TAG", "swan")
SYSTEMS = [("LOTUS", "lotus")]
for name, tag in (("BlendSQL", "blendsql"), ("Palimpzest", "palimpzest")):
    if os.path.exists(os.path.join(HERE, "results", tag + "_agentbench_results.json")):
        SYSTEMS.append((name, tag))
SYSTEMS += [("PLOP", "plop"), ("SWAN", SWAN_TAG)]

ev_src = open(os.path.join(HERE, "eval_agentbench.py")).read().replace("\nmain()\n", "\n")
ev = importlib.util.module_from_spec(importlib.util.spec_from_loader("ev", loader=None))
ev.__dict__["__file__"] = os.path.join(HERE, "eval_agentbench.py")
exec(compile(ev_src, "eval_agentbench.py", "exec"), ev.__dict__)

HEADER = """# agent_bench Q1-Q30: LOTUS vs BlendSQL vs Palimpzest vs PLOP-DP vs SWAN (gpt-5.6-luna) -- {today}

Quality = deterministic scorer (eval_agentbench.py): LIMIT-free PLOP execution as ground truth; LIMIT queries scored sound+complete. Calls = true request counts through the cache proxy. Each system's columns are one full run of the 30 queries ({runs}), a replay of its recorded calls with their recorded response times, one system at a time; SWAN replays PLOP's samples via the shared-verdict alias. LOTUS translations are ours (aisql-bench/README.md, provenance table). SWAN run: results/{swan_tag}.

**Read the LOTUS quality column with care.** SWAN and PLOP consume the *same* recorded verdict for every
shared prompt (the proxy's shared-verdict alias), so they agree with the ground truth by construction
wherever their plans agree. LOTUS uses its own prompt templates, so its verdicts are independent samples of
the same model: on queries whose answer hinges on a handful of judgments ({empty} have an EMPTY
ground truth; LOTUS admitted {lotus_rows} rows) any disagreement scores 0 even when LOTUS's answer
is a perfectly reasonable reading. Its macro therefore measures verdict agreement with PLOP's samples as
much as correctness; the calls / latency / cost columns are the like-for-like comparison. BlendSQL (our
mechanical translation, translate_to_blendsql.py; zero-shot; cost estimated from its token counts) is in
the same position as LOTUS: independent samples. Palimpzest (Abacus optimizer; the LOTUS
programs with their semantic operators served by Palimpzest; cost estimated from its token counts) likewise.

"""


def table_header():
    names = [n for n, _ in SYSTEMS]
    cols = names + [f"{n} calls" for n in names] + [f"{n} lat (s)" for n in names] + [f"{n} $" for n in names]
    return "| q | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)


def run_days(per_query):
    days = sorted({(r.get("ran_at") or "")[:10] for r in per_query.values()} - {""})
    return "undated" if not days else days[0] if len(days) == 1 else f"{days[0]} to {days[-1]}"


def main():
    runs = {name: json.load(open(os.path.join(HERE, "results", tag + "_agentbench_results.json")))["per_query"]
            for name, tag in SYSTEMS}
    # the header's facts are read from the result files, so a regenerated table never cites an older run
    empty = [f"Q{i}" for i in range(1, 31) if not sum(ev.scorer.load_rows(ev.GT_TAG, f"Q{i}")[1].values())]
    lotus_rows = " / ".join(f"{sum(ev.scorer.load_rows('lotus', q)[1].values()):,}" for q in empty)
    lines = [HEADER.format(today=date.today().isoformat(), swan_tag=SWAN_TAG, runs=", ".join(
        f"{name} {run_days(runs[name])}" for name, _ in SYSTEMS), empty="/".join(empty),
        lotus_rows=lotus_rows).rstrip("\n"), "", table_header()]
    q_sum = {n: [] for n, _ in SYSTEMS}
    tot = {n: {"calls": 0, "lat": 0.0, "cost": 0.0} for n, _ in SYSTEMS}
    for i in range(1, 31):
        q = f"Q{i}"
        cells = []
        for name, tag in SYSTEMS:
            v, _ = ev.score(tag, q)
            q_sum[name].append(v or 0.0)
            cells.append(f"{(v or 0.0):.3f}")
        for key, fmt in (("calls", "{:,}"), ("lat", "{:.1f}"), ("cost", "{:.4f}")):
            for name, _ in SYSTEMS:
                r = runs[name].get(q, {"llm_calls": 0, "latency_s": 0.0, "cost_usd": 0.0})  # a system not yet run on q
                val = {"calls": r["llm_calls"], "lat": r["latency_s"],
                       "cost": r.get("cost_usd", r.get("cost_usd_est", 0.0))}[key]
                tot[name][key] += val
                cells.append(fmt.format(val))
        lines.append(f"| {q} | " + " | ".join(cells) + " |")
    macro = " | ".join(f"**{sum(q_sum[n]) / 30:.3f}**" for n, _ in SYSTEMS)
    calls = " | ".join(f"{tot[n]['calls']:,}" for n, _ in SYSTEMS)
    lat = " | ".join(f"{tot[n]['lat']:.0f}" for n, _ in SYSTEMS)
    cost = " | ".join(f"{tot[n]['cost']:.3f}" for n, _ in SYSTEMS)
    lines.append(f"| **macro / Σ** | {macro} | {calls} | {lat} | {cost} |")
    out = "\n".join(lines) + "\n"
    path = os.path.join(HERE, "results", "agentbench_comparison_three_way.md")
    open(path, "w").write(out)
    print(out)
    print(f"written -> {path}")


main()
