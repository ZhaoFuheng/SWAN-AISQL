#!/usr/bin/env python3
"""Build plop_gt_queries/: the 30 agent_bench queries with their OUTPUT-truncating LIMIT removed,
so the ground truth is fully determined.

Why. Three queries end in `LIMIT k` with no ORDER BY, so their result is underdetermined -- any k
rows are correct. Scoring by row identity against one system's arbitrary choice then measures which
rows an engine reached first rather than whether it is right, and penalises an optimizer for being
efficient (agent_bench Q1: classifying 1 book instead of 200 is a 40x saving and returns a
different, equally valid five rows). With the limit gone the ground truth is the full result, and
eval_agentbench.py scores a limited query on what the SQL actually promises -- soundness and
cardinality.

Cost of removing them, measured: Q1 149 calls, Q2 100, Q3 79 (the unlimited runs, $0.037 for all
three). PLOP was already classifying nearly the whole relation before its limit truncated the
output, so the limit-free ground truth costs essentially nothing extra. If a future query made the
unlimited run expensive, the answer would be to keep its limit and score it by exact row
comparison like the other 27.

Only a TRAILING limit is removed. Limits inside subqueries and CTEs define the data set (Q30's
`SELECT DISTINCT l_suppkey ... LIMIT 40`) and changing them would change the question.

  python3 make_gt_queries.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "plop_queries")
DST = os.path.join(HERE, "plop_gt_queries")


def main():
    os.makedirs(DST, exist_ok=True)
    limits = {}
    for i in range(1, 31):
        q = f"Q{i}"
        src = open(os.path.join(SRC, q + ".sql")).read()
        match = re.search(r"(?is)\bLIMIT\s+(\d+)\s*;?\s*$", src.rstrip())
        if not match:
            open(os.path.join(DST, q + ".sql"), "w").write(src)
            continue
        limits[q] = int(match.group(1))
        body = src.rstrip()[: match.start()].rstrip()
        if not body.endswith(";"):
            body += ";"
        open(os.path.join(DST, q + ".sql"), "w").write(body + "\n")
    json.dump(limits, open(os.path.join(DST, "output_limits.json"), "w"), indent=1)
    print("limit removed:", ", ".join(sorted(limits, key=lambda s: int(s[1:]))) or "(none)")
    print("stage into the PLOP tree and run the ground truth with:")
    print("  cp plop_gt_queries/Q*.sql $PLOP_ROOT/test/semantic/agent_bench_gt/")
    print("  PLOP_MODE=none PLOP_QUERY_DIR=test/semantic/agent_bench_gt PLOP_TAG=plop_gt \\")
    print("    python3 plop_agentbench.py")


main()
