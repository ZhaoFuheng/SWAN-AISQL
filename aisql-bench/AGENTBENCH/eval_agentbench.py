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

A query the system did not run is no observation: it is shown as n/a and left out of the macro, which then says how
many queries it covers (report_three_way.py does the same, from the same test: score_agentbench.py's `ran`). A query
that ran and failed scores 0.

  python3 eval_agentbench.py [systems...]      (default: swan plop -- the runner's default tag)
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
    """(score, note) for one query; (None, why) when there is nothing to score."""
    gt_cols, _ = scorer.load_rows(GT_TAG, q)
    if gt_cols is None:
        return None, "no ground truth"
    # A query the system did not run is no observation (n/a, outside the macro), not a wrong answer.
    if not scorer.ran(system, q):
        return None, "not run"
    # A query the system could not run scores 0, whatever file its runner left: a failed query writes an empty
    # result, which read as "ran and returned no rows" and scored 1.000 (BlendSQL Q6-Q9 until 2026-11-20).
    if scorer.failed(system, q):
        return 0.0, "the query failed in this system"
    cols, rows = scorer.load_rows(system, q, cols=gt_cols)
    # A system whose output cannot be PARSED must not score: with cols=None the shared column list
    # is empty, the ground truth projects to an empty multiset, and empty-vs-empty scored 1.000 --
    # a malformed result file read as a perfect answer.
    if cols is None:
        return 0.0, "system output missing or unparseable"
    if not gt_cols:
        _, actual = scorer.load_rows(system, q)
        return (1.0, "both empty") if not actual else (0.0, "gt empty, system returned rows")
    if not cols:
        # an EMPTY system result (no header): no shared column, so both sides would project to empty multisets
        # and score 1.000; judge it against the unprojected ground truth instead
        _, gt_all = scorer.load_rows(GT_TAG, q)
        if gt_all:
            return 0.0, "system returned no rows, the ground truth has rows"

    shared = [c for c in gt_cols if cols and c in cols]
    if cols and not shared:
        # a result with none of the ground truth's columns: projected onto no column both sides are empty, which
        # scored 1.000; it matches only when neither side has a row
        _, gt_all = scorer.load_rows(GT_TAG, q)
        _, actual = scorer.load_rows(system, q)
        return (1.0, "both empty") if not gt_all and not actual else (0.0, "no column in common with the ground truth")
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
    systems = sys.argv[1:] or ["swan", "plop"]
    width = max(len(s) for s in systems) + 2
    header = f"{'q':<5}" + "".join(f"{s:>{width}}" for s in systems) + "   note"
    print(header)
    print("-" * len(header))
    macro, unreadable = {s: [] for s in systems}, {s: [] for s in systems}
    for i in range(1, 31):
        q = f"Q{i}"
        line, note = f"{q:<5}", ""
        for s in systems:
            v, n = score(s, q)
            if v is None:
                line += f"{'n/a':>{width}}"
                continue
            macro[s].append(v)
            if n == "system output missing or unparseable":
                unreadable[s].append(q)
            line += f"{v:>{width}.3f}"
            if q in LIMITS and s == systems[0]:
                note = n
        print(line + ("   " + note if note else ""))
    print("-" * len(header))
    print(f"{'macro':<5}" + "".join(
        (f"{sum(macro[s]) / len(macro[s]):>{width}.3f}" if macro[s] else f"{'n/a':>{width}}") for s in systems))
    for s in systems:
        if scorer.coverage_note(s, len(macro[s])):
            print(scorer.coverage_note(s, len(macro[s])))
        if unreadable[s]:
            print(f"  {s}: no readable rows file for {', '.join(unreadable[s])} (results/{s}/<query>.csv or .csv.gz): scored 0")
        total = sum(v.get("llm_calls", 0) for v in (scorer.per_query(s) or {}).values())
        if total:
            print(f"  {s}: {total:,} calls")


main()
