#!/usr/bin/env python3
"""Quality scoring for the agent_bench three-way comparison, per the PLOP paper's protocol:
the un-rewritten UDF execution (PLOP_MODE=none) is the ground truth, and each method's result
set is scored against it by row-multiset F1.

Rows are compared on the columns COMMON to both frames (case-insensitive names), values
normalized (strings stripped/lowercased, numbers rounded to 4 significant places, timestamps
to seconds) so dialect formatting differences don't count as errors. Duplicate rows count with
multiplicity. Empty-vs-empty scores 1.0; empty-vs-nonempty scores 0.0.

eval_agentbench.py is the scorer behind the published table: it reuses the row loading and F1 below and
scores the three LIMIT queries on soundness and completeness. Run directly, this file compares every query
by exact rows against GT_SYSTEM (default plop_gt, the LIMIT-free ground truth).

  python3 score_agentbench.py [systems...]   (default: swan plop lotus)

A query a system did not run is shown as n/a and left out of its macro (`ran` below, which eval_agentbench.py and
report_three_way.py use too); a query that ran and failed scores 0.
"""
import json
import math
import os
import sys
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.environ.get("GT_SYSTEM", "plop_gt")


def norm_val(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.4g}"
    s = str(v).strip().lower()
    try:
        f = float(s)
        return norm_val(f)
    except ValueError:
        pass
    if len(s) >= 10 and s[:2] == "20" and "-" in s[:8]:
        s = s.replace("t", " ")[:19]  # timestamp: unify ISO-T vs space, second precision
    return s


def load_rows(system, q, cols=None):
    p = os.path.join(HERE, "results", system, q + ".csv")
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        p += ".gz"  # bulk outputs (Q19: 52MB) are committed gzipped; pandas reads .csv.gz transparently
    if not os.path.exists(p):
        return None, Counter()
    if os.path.getsize(p) <= 1:
        return [], Counter()  # ran and produced an EMPTY result set (vs None = no result file)
    try:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
    except Exception:
        return None, Counter()
    df.columns = [c.strip().lower() for c in df.columns]
    # SELECT-* queries emit duplicate column names; pandas mangles repeats to name.1/name.2 (and
    # different systems order the copies differently). Keep the FIRST occurrence of each base name
    # so both sides compare the same logical columns.
    import re as _re
    base = [_re.sub(r"\.\d+$", "", c) for c in df.columns]
    keep_idx, seen = [], set()
    for i, b in enumerate(base):
        if b not in seen:
            seen.add(b)
            keep_idx.append(i)
    df = df.iloc[:, keep_idx]
    df.columns = [base[i] for i in keep_idx]
    if cols is not None:
        keep = [c for c in cols if c in df.columns]
        if not keep:
            return list(df.columns), Counter()
        df = df[keep]
    rows = Counter(tuple(norm_val(v) for v in row) for row in df.itertuples(index=False))
    return list(df.columns), rows


def f1(gt, pred):
    if not gt and not pred:
        return 1.0
    inter = sum((gt & pred).values())
    p = inter / sum(pred.values()) if pred else 0.0
    r = inter / sum(gt.values()) if gt else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


_SUMMARIES = {}


def per_query(system):
    """The runner's record of every query it ran (results/<system>_agentbench_results.json; a failed query is listed
    with its error); None when the system has no summary."""
    if system not in _SUMMARIES:
        path = os.path.join(HERE, "results", system + "_agentbench_results.json")
        _SUMMARIES[system] = json.load(open(path)).get("per_query", {}) if os.path.exists(path) else None
    return _SUMMARIES[system]


def ran(system, q):
    """Whether the system ran this query: its runner's summary lists it, or, for a results directory without a
    summary, its output file exists. A query that was not run is no observation: every scoring command shows it as
    n/a and leaves it out of the macro, where a query that ran and failed scores 0."""
    records = per_query(system)
    if records is not None:
        return q in records
    p = os.path.join(HERE, "results", system, q + ".csv")
    return os.path.exists(p) or os.path.exists(p + ".gz")


def failed(system, q):
    """Whether the runner recorded an error for this query: it scores 0, whatever file the runner left."""
    return bool(((per_query(system) or {}).get(q) or {}).get("error"))


def coverage_note(system, scored, total=30):
    """The line a scoring command prints under the macro of a system it did not score on every query."""
    if scored == total:
        return None
    if not scored:
        return f"  {system}: no query scored (nothing under results/{system}/ or in its summary)"
    return (f"  {system}: {scored} of {total} queries scored, the rest n/a (not run, or no ground truth); its macro "
            f"covers the scored ones only and is not comparable with a full run's")


def main():
    systems = sys.argv[1:] or ["swan", "plop", "lotus"]
    queries = [f"Q{i}" for i in range(1, 31)]
    print(f"ground truth: {GT}")
    header = f"{'q':<4}" + "".join(f"{s:>10}" for s in systems)
    print(header)
    macro = {s: [] for s in systems}
    for q in queries:
        gt_cols, gt_rows = load_rows(GT, q)
        line = f"{q:<4}"
        for s in systems:
            if gt_cols is None or not ran(s, q):
                line += f"{'n/a':>10}"
                continue
            cols, rows = load_rows(s, q, cols=gt_cols)
            if failed(s, q):
                score = 0.0
            elif not gt_cols:
                # Ground truth is EMPTY: agreement iff the system is empty too -- judged on the
                # system's UNPROJECTED rows, since projecting onto an empty column list returns an
                # empty Counter for ANY result (which scored every answer 1.0 here).
                _, actual = load_rows(s, q)
                score = 1.0 if not actual else 0.0
            else:
                # re-project ground truth onto the shared columns for a fair multiset match
                shared = [c for c in gt_cols if cols and c in cols]
                # no shared column (an empty or failed result, or one with other columns): compare the unprojected
                # rows of both sides, or empty-vs-empty would score a missing answer 1.0
                if shared:
                    score = f1(load_rows(GT, q, cols=shared)[1], rows)
                else:
                    score = 1.0 if not gt_rows and not load_rows(s, q)[1] else 0.0
            macro[s].append(score)
            line += f"{score:>10.3f}"
        print(line)
    print(f"{'MACRO':<4}" + "".join(
        (f"{sum(macro[s]) / len(macro[s]):>10.3f}" if macro[s] else f"{'n/a':>10}") for s in systems))
    for s in systems:
        if coverage_note(s, len(macro[s])):
            print(coverage_note(s, len(macro[s])))


main()
