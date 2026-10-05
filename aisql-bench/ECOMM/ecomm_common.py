"""Shared scoring + ground-truth helpers for the ECOMM benchmark (lotus_ecomm.py / swan_ecomm.py).

Metrics follow the SemBench TOMLs: f1-score over the output id set for most queries;
adjusted-rand-index for the categorization queries q3-q6 (clustering agreement over the ids
present in BOTH prediction and gold -- label names don't matter, the partition does).
"""
import json
import os
from collections import Counter
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(HERE, "ground_truth")
ARI_QUERIES = {"q3", "q4", "q5", "q6"}


def gt(name):
    return json.load(open(os.path.join(GT_DIR, name + ".json")))["ground_truth"]


def norm(s):
    return " ".join(str(s).strip().lower().split())


def prf(out, gold):
    O, G = {norm(x) for x in out}, {norm(x) for x in gold}
    tp = len(O & G)
    P = tp / len(O) if O else (1.0 if not G else 0.0)
    R = tp / len(G) if G else 1.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return round(P, 3), round(R, 3), round(F, 3)


def ari(pred_pairs, gold_pairs):
    """Adjusted Rand Index between two (id, category) assignments, over ids present in both."""
    gp = {norm(i): norm(c) for i, c in gold_pairs}
    pp = {norm(i): norm(c) for i, c in pred_pairs}
    ids = [i for i in gp if i in pp]
    n = len(ids)
    if n < 2:
        return 0.0
    a = [gp[i] for i in ids]
    b = [pp[i] for i in ids]
    sum_ab = sum(comb(v, 2) for v in Counter(zip(a, b)).values())
    sum_a = sum(comb(v, 2) for v in Counter(a).values())
    sum_b = sum(comb(v, 2) for v in Counter(b).values())
    exp = sum_a * sum_b / comb(n, 2)
    mx = (sum_a + sum_b) / 2
    return round((sum_ab - exp) / (mx - exp), 3) if mx != exp else 1.0


def score(name, out, gold):
    """Returns (precision, recall, main_metric). For ARI queries P/R are coverage diagnostics."""
    if name in ARI_QUERIES:
        pred_ids = {norm(i) for i, _ in out}
        gold_ids = {norm(i) for i, _ in gold}
        cov = len(pred_ids & gold_ids)
        P = cov / len(pred_ids) if pred_ids else 0.0
        R = cov / len(gold_ids) if gold_ids else 1.0
        return round(P, 3), round(R, 3), ari(out, gold)
    return prf(out, gold)
