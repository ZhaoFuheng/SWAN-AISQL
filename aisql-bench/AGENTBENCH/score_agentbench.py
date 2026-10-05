#!/usr/bin/env python3
"""Quality scoring for the agent_bench three-way comparison, per the PLOP paper's protocol:
the un-rewritten UDF execution (PLOP_MODE=none) is the ground truth, and each method's result
set is scored against it by row-multiset F1.

Rows are compared on the columns COMMON to both frames (case-insensitive names), values
normalized (strings stripped/lowercased, numbers rounded to 4 significant places, timestamps
to seconds) so dialect formatting differences don't count as errors. Duplicate rows count with
multiplicity. Empty-vs-empty scores 1.0; empty-vs-nonempty scores 0.0.

  python3 score_agentbench.py [systems...]   (default: swan plop lotus, vs plop_none)
"""
import json
import math
import os
import sys
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.environ.get("GT_SYSTEM", "plop_none")


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
            if gt_cols is None:
                line += f"{'n/a':>10}"
                continue
            cols, rows = load_rows(s, q, cols=gt_cols)
            if not gt_cols:
                # Ground truth is EMPTY: agreement iff the system is empty too -- judged on the
                # system's UNPROJECTED rows, since projecting onto an empty column list returns an
                # empty Counter for ANY result (which scored every answer 1.0 here).
                _, actual = load_rows(s, q)
                score = 1.0 if not actual else 0.0
            else:
                # re-project ground truth onto the shared columns for a fair multiset match
                shared = [c for c in gt_cols if cols and c in cols]
                _, gt_shared = load_rows(GT, q, cols=shared) if shared else (None, Counter())
                score = f1(gt_shared, rows)
            macro[s].append(score)
            line += f"{score:>10.3f}"
        print(line)
    print(f"{'MACRO':<4}" + "".join(
        f"{(sum(macro[s]) / len(macro[s]) if macro[s] else float('nan')):>10.3f}" for s in systems))


main()
