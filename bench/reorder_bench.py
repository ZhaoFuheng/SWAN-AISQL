#!/usr/bin/env python3
"""Reorder benchmark over 18 govreport queries, reorder OFF vs ON (DUCKDB_AI_REORDER), on the same build.
Reports chat_calls + latency per query; result id-set must match (reordering is answer-preserving).

  9 "filter" queries -- the previously-selected ai_filter boolean trees (conj/disj/mix, 2/6/10 leaves).
  9 "mixed"  queries -- 2-3 AI conditions combining ai_filter / ai_classify(..)=x / ai_score(..)>v /
                        ai_complete(..)=x in the WHERE clause (exercises the generalized reorder).

Env: BENCH_LIMIT (docs; default 100 to keep the new uncached mixed-query prompts cheap)  BENCH_PROXY.
Reuses bench.py's LOAD / _duck / run / measure_overhead helpers.
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("BENCH_LIMIT", "100")  # mixed queries hit new prompts; keep it modest by default
spec = importlib.util.spec_from_file_location("b", f"{HERE}/bench.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

CATS = "['health','defense','economy','education','environment','technology','justice','other']"


def S(txt):  # ai_score criterion helper
    return f"ai_score(summary, '{txt}')"


def C(cat):  # ai_classify(..)=cat helper
    return f"ai_classify(summary, {CATS}) = '{cat}'"


def F(q):  # ai_filter helper
    return f"ai_filter('{q}: ' || summary)"


def M(instr, val):  # ai_complete(..)=val helper
    return f"ai_complete('{instr}: ' || summary) = '{val}'"


def C_IN(cats):  # ai_classify(..) IN (...) -- set-membership leaf
    vals = ", ".join(f"'{c}'" for c in cats)
    return f"ai_classify(summary, {CATS}) IN ({vals})"


def C_NOTIN(cats):  # ai_classify(..) NOT IN (...)
    vals = ", ".join(f"'{c}'" for c in cats)
    return f"ai_classify(summary, {CATS}) NOT IN ({vals})"


# 11 mixed-AI predicates (2-3 conditions each, mixing filter/classify/score/complete); the last 2 use a
# set-membership IN / NOT IN on ai_classify.
MIXED = [
    ("mix1_F_C", f"{F('Does this report discuss public health or medical services')} AND {C('health')}"),
    ("mix2_C_S", f"{C('defense')} AND {S('relevance to national security')} > 0.5"),
    ("mix3_S_F", f"{S('economic significance')} > 0.5 AND {F('Does this report discuss the federal budget')}"),
    ("mix4_F_S", f"{F('Is this a technical or scientific report')} AND {S('technical complexity')} > 0.6"),
    ("mix5_C_F_S", f"{C('health')} AND {F('Does it mention federal funding')} AND {S('overall importance')} > 0.4"),
    ("mix6_S_C", f"{S('public importance')} > 0.7 AND {C('environment')}"),
    ("mix7_F_C_S", f"{F('Does this involve government oversight')} AND {C('defense')} AND {S('urgency')} > 0.5"),
    ("mix8_C_or_F", f"{C('education')} OR {F('Does this report discuss schools or universities')}"),
    ("mix9_M_F", f"{M('In one word, the primary topic of this report', 'defense')} AND "
                 f"{F('Is this a defense-related report')}"),
    ("set1_Cin_F", f"{C_IN(['health', 'defense'])} AND {F('Does this report discuss federal policy')}"),
    ("set2_Cnotin_S", f"{C_NOTIN(['other', 'justice'])} AND {S('public importance')} > 0.5"),
]


def selected_filter_queries():
    manifest = json.load(open(f"{HERE}/manifest.json"))
    sel = json.load(open(f"{HERE}/selected.json"))
    out = []
    for setn in ("conjunction", "disjunction", "mix"):
        by = {e["q"]: e for e in manifest[setn]}
        for q in sel[setn]:
            out.append((f"{setn[:4]}_q{q}_{by[q]['n_leaves']}L", by[q]["sql"]))
    return out


def main():
    if not os.path.exists(b.BIN):
        sys.exit(f"duckdb not found at {b.BIN}")
    queries = [("filter", n, w) for n, w in selected_filter_queries()] + [("mixed", n, w) for n, w in MIXED]
    docs = b._duck(f"{b.LOAD} SELECT count(*) AS n FROM {b.TABLE};")[-1]["n"]
    oh = b.measure_overhead()
    sys.stderr.write(f"docs={docs} queries={len(queries)} (9 filter + 11 mixed, incl. 2 set-op)\n")
    results = []
    for kind, name, where in queries:
        off = b.run(where, reorder=False, overhead=oh)
        on = b.run(where, reorder=True, overhead=oh)
        results.append((kind, name, off, on))
        sys.stderr.write(f"[{name}] {kind}  calls {off['chat_calls']}->{on['chat_calls']}  "
                         f"lat {off['latency_s']:.1f}->{on['latency_s']:.1f}s  match={off['checksum']==on['checksum']}\n")
        sys.stderr.flush()
    report(docs, results)
    json.dump([{"kind": k, "name": n, "off": o, "on": x} for k, n, o, x in results],
              open(f"{HERE}/reorder_bench_results.json", "w"), indent=1)


def pct(a, bb):
    return 0.0 if a == 0 else 100.0 * (a - bb) / a


def report(docs, results):
    print(f"\n=== reorder OFF vs ON, 18 queries (docs={docs}, gpt-5-mini) ===")
    print(f"{'query':<18} {'kind':<7} {'rows':<6} {'calls_off':<10} {'calls_on':<9} {'d%':<5} "
          f"{'lat_off':<8} {'lat_on':<8} {'match'}")
    print("-" * 84)
    tot = {"filter": [0, 0], "mixed": [0, 0]}
    for kind, name, off, on in results:
        tot[kind][0] += off["chat_calls"]
        tot[kind][1] += on["chat_calls"]
        print(f"{name:<18} {kind:<7} {off['n']:<6} {off['chat_calls']:<10} {on['chat_calls']:<9} "
              f"{pct(off['chat_calls'], on['chat_calls']):<5.0f} {off['latency_s']:<8.1f} {on['latency_s']:<8.1f} "
              f"{off['checksum']==on['checksum']}")
    print("-" * 84)
    for kind in ("filter", "mixed"):
        o, n = tot[kind]
        print(f"{kind} total: calls {o} -> {n} ({pct(o, n):+.0f}%)")
    print(f"\nwrote {HERE}/reorder_bench_results.json")


if __name__ == "__main__":
    main()
