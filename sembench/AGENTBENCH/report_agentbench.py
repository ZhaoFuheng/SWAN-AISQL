#!/usr/bin/env python3
"""Q1-Q30 SWAN-vs-PLOP comparison report: quality (row-multiset F1 vs the PLOP-none ground
truth, per the paper's protocol), LLM calls, and latency, from the recorded runs.

Reads results/{swan,plop}_agentbench_results.json for calls/latency and reuses the scorer's
row-matching (shared columns, normalized values, multiplicity) for quality. Prints a markdown
table and writes results/agentbench_comparison.md.

  python3 report_agentbench.py
"""
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# reuse the scorer's load_rows/f1 so quality here always matches score_agentbench.py
spec = importlib.util.spec_from_file_location("scorer", os.path.join(HERE, "score_agentbench.py"))
scorer_src = open(os.path.join(HERE, "score_agentbench.py")).read()
scorer_src = scorer_src.replace("\nmain()\n", "\n")  # import without executing the CLI
scorer = importlib.util.module_from_spec(spec)
exec(compile(scorer_src, "score_agentbench.py", "exec"), scorer.__dict__)

GT = "plop_none"
# which recorded SWAN run to report on (results/<tag>/ + results/<tag>_agentbench_results.json)
SWAN_TAG = os.environ.get("SWAN_TAG", "swan")


def quality(system, q):
    gt_cols, _ = scorer.load_rows(GT, q)
    if gt_cols is None:
        return None  # ground truth missing entirely
    cols, rows = scorer.load_rows(system, q, cols=gt_cols)
    if not gt_cols:
        # Empty ground truth: agreement iff the system is empty too. Judge that on the system's
        # UNPROJECTED rows -- projecting onto the GT's (empty) column list takes load_rows' "no
        # shared columns" shortcut, which returns an empty Counter whatever the system produced,
        # so testing `rows` here scored every answer 1.0 (agent_bench Q30: 2 rows vs an empty GT).
        _, actual = scorer.load_rows(system, q)
        return 1.0 if not actual else 0.0
    shared = [c for c in gt_cols if cols and c in cols]
    from collections import Counter
    _, gt_shared = scorer.load_rows(GT, q, cols=shared) if shared else (None, Counter())
    return scorer.f1(gt_shared, rows)


def main():
    swan = json.load(open(os.path.join(HERE, "results", SWAN_TAG + "_agentbench_results.json")))["per_query"]
    plop = json.load(open(os.path.join(HERE, "results", "plop_agentbench_results.json")))["per_query"]
    lines = [
        "# agent_bench: SWAN vs PLOP-DP (gpt-5.6-luna, shared-verdict protocol)",
        "",
        "Quality = row-multiset F1 vs the un-rewritten PLOP execution (the paper's ground",
        "truth); calls = true request counts via the recording proxy (query-scoped caching);",
        "latency includes recorded-latency replay for cached calls.",
        "",
        "| q | SWAN F1 | PLOP F1 | SWAN calls | PLOP calls | SWAN lat (s) | PLOP lat (s) | SWAN $ | PLOP $ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    sums = {"sq": [], "pq": [], "sc": 0, "pc": 0, "sl": 0.0, "pl": 0.0, "s$": 0.0, "p$": 0.0}
    for i in range(1, 31):
        q = f"Q{i}"
        s, p = swan[q], plop[q]
        sq, pq = quality(SWAN_TAG, q), quality("plop", q)
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        scost = s.get("cost_usd", s.get("cost_usd_est", 0.0))
        pcost = p.get("cost_usd", p.get("cost_usd_est", 0.0))
        lines.append(f"| {q} | {fmt(sq)} | {fmt(pq)} | {s['llm_calls']:,} | {p['llm_calls']:,} "
                     f"| {s['latency_s']:.1f} | {p['latency_s']:.1f} | {scost:.4f} | {pcost:.4f} |")
        sums["s$"] += scost
        sums["p$"] += pcost
        if sq is not None:
            sums["sq"].append(sq)
            sums["pq"].append(pq)
        sums["sc"] += s["llm_calls"]
        sums["pc"] += p["llm_calls"]
        sums["sl"] += s["latency_s"]
        sums["pl"] += p["latency_s"]
    lines.append(f"| **macro/Σ** | **{sum(sums['sq'])/len(sums['sq']):.3f}** "
                 f"| **{sum(sums['pq'])/len(sums['pq']):.3f}** | **{sums['sc']:,}** "
                 f"| **{sums['pc']:,}** | **{sums['sl']:.1f}** | **{sums['pl']:.1f}** "
                 f"| **{sums['s$']:.4f}** | **{sums['p$']:.4f}** |")
    out = "\n".join(lines) + "\n"
    suffix = "" if SWAN_TAG == "swan" else "_" + SWAN_TAG
    path = os.path.join(HERE, "results", "agentbench_comparison" + suffix + ".md")
    open(path, "w").write(out)
    print(out)
    print(f"written -> {path}")


main()
