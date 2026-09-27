#!/usr/bin/env python3
"""Q1-Q30 three-way table: SWAN vs PLOP-DP vs LOTUS, scored by the deterministic scorer
(eval_agentbench.py: LIMIT-free PLOP ground truth, LIMIT queries sound+complete).

Reads results/{<SWAN_TAG>,plop,lotus}_agentbench_results.json for calls / latency / cost and writes
results/agentbench_comparison_three_way.md.

  SWAN_TAG=swan_fix3 python3 report_three_way.py
"""
import importlib.util
import json
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
SWAN_TAG = os.environ.get("SWAN_TAG", "swan")
SYSTEMS = [("SWAN", SWAN_TAG), ("PLOP", "plop"), ("LOTUS", "lotus")]

ev_src = open(os.path.join(HERE, "eval_agentbench.py")).read().replace("\nmain()\n", "\n")
ev = importlib.util.module_from_spec(importlib.util.spec_from_loader("ev", loader=None))
ev.__dict__["__file__"] = os.path.join(HERE, "eval_agentbench.py")
exec(compile(ev_src, "eval_agentbench.py", "exec"), ev.__dict__)

HEADER = """# agent_bench Q1-Q30: SWAN vs PLOP-DP vs LOTUS (gpt-5.6-luna) -- {today}

Quality = deterministic scorer (eval_agentbench.py): LIMIT-free PLOP execution as ground truth; LIMIT queries scored sound+complete. Calls = true request counts through the recording proxy (SWAN replays PLOP's samples via the shared-verdict alias; LOTUS Q1-Q18 replayed from 2026-09-21, Q19-Q30 recorded fresh 2026-09-26). LOTUS translations are ours (sembench/README.md, provenance table). SWAN run: results/{swan_tag}.

**Read the LOTUS quality column with care.** SWAN and PLOP consume the *same* recorded verdict for every
shared prompt (the proxy's shared-verdict alias), so they agree with the ground truth by construction
wherever their plans agree. LOTUS uses its own prompt templates, so its verdicts are independent samples of
the same model: on queries whose answer hinges on a handful of judgments (Q13/Q16/Q17/Q30 have an EMPTY
ground truth; LOTUS admitted 3 / 261 / 3,026 / 4 rows) any disagreement scores 0 even when LOTUS's answer
is a perfectly reasonable reading. Its macro therefore measures verdict agreement with PLOP's samples as
much as correctness; the calls / latency / cost columns are the like-for-like comparison.

| q | SWAN | PLOP | LOTUS | SWAN calls | PLOP calls | LOTUS calls | SWAN lat (s) | PLOP lat (s) | LOTUS lat (s) | SWAN $ | PLOP $ | LOTUS $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""


def main():
    runs = {name: json.load(open(os.path.join(HERE, "results", tag + "_agentbench_results.json")))["per_query"]
            for name, tag in SYSTEMS}
    lines = [HEADER.format(today=date.today().isoformat(), swan_tag=SWAN_TAG).rstrip("\n")]
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
                r = runs[name][q]
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
