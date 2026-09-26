#!/usr/bin/env python3
"""Score a system against the DETERMINISTIC ground truth (plop_gt), which is PLOP's un-rewritten
execution with the output-truncating LIMIT removed.

Why this exists. Three of the thirty queries end in `LIMIT k` with no ORDER BY, so their result is
UNDERDETERMINED: any k rows are a correct answer. Scoring them by row equality against one system's
arbitrary choice measures which rows an engine happened to reach first, not whether it is right --
and it penalises an optimizer for being efficient. agent_bench Q1: classifying one book instead of
all 200 is a 40x call saving and returns a different, equally valid five rows.

So a limited query is scored on what the SQL actually promises:
  * SOUND    -- every row returned appears in the unlimited ground truth
  * COMPLETE -- exactly min(k, |ground truth|) rows are returned
Both hold => 1.0. Unlimited queries keep exact row-multiset F1, which is the right measure there.

  python3 eval_agentbench.py [systems...]      (default: swan_final plop)
  env: GT_TAG (default plop_gt)
"""
import importlib.util
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
GT_TAG = os.environ.get("GT_TAG", "plop_gt")

# reuse the scorer's row loading and normalisation so "same row" means the same thing everywhere
scorer_src = open(os.path.join(HERE, "score_agentbench.py")).read().replace("\nmain()\n", "\n")
scorer = importlib.util.module_from_spec(importlib.util.spec_from_loader("scorer", loader=None))
scorer.__dict__["__file__"] = os.path.join(HERE, "score_agentbench.py")
exec(compile(scorer_src, "score_agentbench.py", "exec"), scorer.__dict__)

LIMITS = json.load(open(os.path.join(HERE, "plop_gt_queries", "output_limits.json")))


def score(system, q):
    """(score, note) for one query."""
    gt_cols, _ = scorer.load_rows(GT_TAG, q)
    if gt_cols is None:
        return None, "no ground truth"
    cols, rows = scorer.load_rows(system, q, cols=gt_cols)
    # A system whose output cannot be PARSED must not score: with cols=None the shared column list
    # is empty, the ground truth projects to an empty multiset, and empty-vs-empty scored 1.000 --
    # a malformed result file read as a perfect answer.
    if cols is None:
        return 0.0, "system output missing or unparseable"
    if not gt_cols:
        _, actual = scorer.load_rows(system, q)
        return (1.0, "both empty") if not actual else (0.0, "gt empty, system returned rows")

    shared = [c for c in gt_cols if cols and c in cols]
    _, gt_rows = scorer.load_rows(GT_TAG, q, cols=shared) if shared else (None, Counter())

    k = LIMITS.get(q)
    if k is None:
        return scorer.f1(gt_rows, rows), "exact"

    # LIMITED query: soundness + cardinality, never row identity
    unsound = sum(n for row, n in (rows - gt_rows).items())
    expected = min(k, sum(gt_rows.values()))
    got = sum(rows.values())
    if unsound:
        return 0.0, f"{unsound} row(s) not in the ground truth"
    if got != expected:
        return 0.0, f"returned {got} rows, expected min({k}, {sum(gt_rows.values())}) = {expected}"
    return 1.0, f"sound + complete ({got} of {sum(gt_rows.values())}, limit {k})"


def main():
    systems = sys.argv[1:] or ["swan_final", "plop"]
    calls = {}
    for s in systems:
        path = os.path.join(HERE, "results", s + "_agentbench_results.json")
        calls[s] = json.load(open(path))["per_query"] if os.path.exists(path) else {}

    width = max(len(s) for s in systems) + 2
    header = f"{'q':<5}" + "".join(f"{s:>{width}}" for s in systems) + "   note"
    print(header)
    print("-" * len(header))
    macro = {s: [] for s in systems}
    for i in range(1, 31):
        q = f"Q{i}"
        line, note = f"{q:<5}", ""
        for s in systems:
            v, n = score(s, q)
            if v is None:
                line += f"{'n/a':>{width}}"
                continue
            macro[s].append(v)
            line += f"{v:>{width}.3f}"
            if q in LIMITS and s == systems[0]:
                note = n
        print(line + ("   " + note if note else ""))
    print("-" * len(header))
    print(f"{'macro':<5}" + "".join(
        f"{(sum(macro[s]) / len(macro[s]) if macro[s] else float('nan')):>{width}.3f}" for s in systems))
    for s in systems:
        total = sum(v.get("llm_calls", 0) for v in calls[s].values()) if calls[s] else 0
        if total:
            print(f"  {s}: {total:,} calls")


main()
