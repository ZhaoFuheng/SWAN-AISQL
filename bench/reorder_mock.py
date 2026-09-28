#!/usr/bin/env python3
"""Call-count-only reorder comparison on the govreport selected queries, against the in-process mock
(deterministic verdict per prompt, $0, minutes). Per query, three fresh processes:
  OFF        SET ai_reorder=false            -- written order, no DP
  ON-tuple   ai_factorize='off'               -- the node on the scalar path: tuple-batch per-row JIT DP (July baseline's algorithm)
  ON-leaf    default                          -- the per-leaf region (new)
The id-set checksum must agree across all three (result-preserving); the calls are the comparison.
CAVEAT: mock verdicts are hash-based, so the selectivity model cannot learn them -- on the mock neither
DP variant learns, and this harness therefore CANNOT detect a JIT-learning regression. It checks the
ordering machinery and result preservation. For the learned ordering, replay real recordings
(conj_q0 / conj_q44 through the :4001 proxy) and compare calls against reorder_bench_results.json.
  python3 reorder_mock.py [conj_q44_10L ...]"""
import json, os, subprocess, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.abspath(os.path.join(HERE, "../build/release/duckdb"))
manifest = json.load(open(f"{HERE}/manifest.json")); selected = json.load(open(f"{HERE}/selected.json"))
LOAD = f"CREATE TABLE govreport AS SELECT id, summary, source FROM read_csv('{HERE}/govreport_summary.csv', header=true);"
VARIANTS = [("OFF", "SET ai_reorder=false;"), ("ON-tuple", "SET ai_factorize='off';"), ("ON-leaf", "")]

def run(sql, setting):
    script = f"CALL ai_mock_start(); {setting}\n{LOAD}\nSELECT ai_usage_reset();\n.mode json\nCREATE TABLE r AS {sql}\nSELECT md5(string_agg(id::VARCHAR, ',' ORDER BY id)) AS ids FROM r;\nSELECT (sum(llm_calls)-sum(embed_calls))::BIGINT AS calls FROM ai_usage();\n"
    env = {**os.environ, "AI_MODEL": "mock", "AI_API_KEY": "x", "AI_MAX_CONCURRENCY": "20"}
    p = subprocess.run([BIN], input=script, env=env, capture_output=True, text=True, timeout=3600)
    if p.returncode != 0:
        return None, "ERR " + p.stderr.strip().splitlines()[-1][:120] if p.stderr.strip() else "ERR"
    objs = []
    for l in p.stdout.splitlines():
        if l.startswith("[") or l.startswith("{"):
            try: objs.append(json.loads(l))
            except json.JSONDecodeError: pass
    ids = next((o[0]["ids"] for o in objs if isinstance(o, list) and o and "ids" in o[0]), None)
    calls = next((o[0]["calls"] for o in objs if isinstance(o, list) and o and "calls" in o[0]), None)
    return calls, (ids or 'NONE')[:8]

names = sys.argv[1:] or [f"{s[:4]}_q{q}_{[e for e in manifest[s] if e['q']==q][0]['n_leaves']}L" for s in ("conjunction","disjunction","mix") for q in selected[s]]
print(f"{'query':16}{'OFF':>7}{'ON-tuple':>10}{'ON-leaf':>9}{'leaf vs tuple':>15}  checksums")
tot = {"OFF":0,"ON-tuple":0,"ON-leaf":0}
for name in names:
    setname = {"conj":"conjunction","disj":"disjunction","mix":"mix"}[name.split("_")[0]]
    q = int(name.split("_")[1][1:]); sql = [e for e in manifest[setname] if e["q"]==q][0]["sql"]
    if not sql.strip().upper().startswith("SELECT"): sql = f"SELECT id FROM govreport WHERE {sql}"
    sql = sql.rstrip().rstrip(";") + ";"
    res = {v: run(sql, s) for v, s in VARIANTS}
    calls = {v: res[v][0] for v in res}; sums = {v: res[v][1] for v in res}
    for v in tot: tot[v] += calls[v] or 0
    agree = "match" if len(set(sums.values())) == 1 else "MISMATCH"
    lt = f"{(calls['ON-leaf']-calls['ON-tuple'])/calls['ON-tuple']*100:+.1f}%" if calls['ON-tuple'] else "-"
    print(f"{name:16}{calls['OFF']:>7}{calls['ON-tuple']:>10}{calls['ON-leaf']:>9}{lt:>15}  {agree} {sums}")
print("-"*90); print(f"{'TOTAL':16}{tot['OFF']:>7}{tot['ON-tuple']:>10}{tot['ON-leaf']:>9}{(tot['ON-leaf']-tot['ON-tuple'])/max(tot['ON-tuple'],1)*100:>14.1f}%")
