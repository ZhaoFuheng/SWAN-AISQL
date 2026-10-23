#!/usr/bin/env python3
"""Prefix-reuse bench: each pattern of patterns.sql over GovReport documents and ECOMM images, with the
provider prefix cache ON and OFF, reporting calls, written and cached tokens and cost from ai_usage().

  python3 run.py                       # live through the proxy (cents): the cost saving per pattern
  python3 run.py --mock                # deterministic counts under the mock (no provider)
  env: DUCKDB_BIN, AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna), DOCS (14), PICS (40), DOC_CHARS (8000)
  Each pattern runs on its own slice of the documents and images (7 x DOCS <= 100 reports).

Documents: govreport_sample.csv (100 GovReport reports, truncated to DOC_CHARS so each is ~2k tokens, over the
NOT COMMITTED (5.5 MB): regenerate it next to this script with the `datasets` package --
    from datasets import load_dataset; import csv
    rows = load_dataset("ccdv/govreport-summarization", split="train[:100]")
    with open("govreport_sample.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "report", "summary"])
        for i, r in enumerate(rows): w.writerow([i, r["report"], r["summary"]])
provider's 1,024-token minimum). Images: aisql-bench/ECOMM/data/sf_500/images. Prefix cache off = the cost
baseline: the same calls, nothing written or read.
"""
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BIN = os.environ.get("DUCKDB_BIN", os.path.join(ROOT, "build", "reldebug", "duckdb"))
MOCK = "--mock" in sys.argv
DOCS = int(os.environ.get("DOCS", "14")); PICS = int(os.environ.get("PICS", "40")); DOC_CHARS = int(os.environ.get("DOC_CHARS", "8000"))
ECOMM = os.path.join(ROOT, "aisql-bench", "ECOMM")
ENV = {**os.environ, "AI_PROXY_URL": os.environ.get("AI_PROXY_URL", "http://localhost:4001"),
       "AI_MODEL": os.environ.get("AI_MODEL", "gpt-5.6-luna"), "AI_API_KEY": "sk-test",
       "AI_MAX_CONCURRENCY": os.environ.get("AI_MAX_CONCURRENCY", "20"), "AI_TIMEOUT_MS": "180000"}

def patterns():
    text = open(os.path.join(HERE, "patterns.sql")).read()
    out = []
    for block in re.split(r"\n(?=-- P\d)", text):
        m = re.match(r"-- (P\d) ([^\n]*)\n(.*)", block, re.S)
        if m:
            out.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
    return out

def setup(slot=0):
    # each pattern gets its own documents and images (slot = pattern index): a span one pattern wrote stays
    # held at the provider for 30 minutes, and a later pattern over the same inputs would read it for free
    # and overstate its own saving
    sql = []
    if MOCK:
        sql.append("CALL ai_mock_start();")
    sql.append(f"CREATE TABLE docs AS SELECT id, substr(report, 1, {DOC_CHARS}) AS report FROM read_csv('{HERE}/govreport_sample.csv', header=true) ORDER BY id LIMIT {DOCS} OFFSET {slot * DOCS};")
    sql.append("CREATE TABLE questions AS SELECT * FROM (VALUES ('Does it cite a law?'), ('Does it mention a deadline?'), ('Does it quantify a risk?'), ('Does it name a state?')) q(q);")
    sql.append(f"ATTACH '{ECOMM}/ecomm.db' AS e (READ_ONLY);")
    sql.append(f"CREATE TABLE pics AS SELECT i.id, '{ECOMM}/' || i.filepath AS filepath FROM e.images i ORDER BY i.id LIMIT {PICS} OFFSET {slot * PICS};")
    sql.append("SET ai_hedge = false;")
    return "\n".join(sql)

def run(sql_body, prefix_on, slot):
    script = setup(slot) + f"\nSET ai_prefix_cache = {'true' if prefix_on else 'false'};\nSELECT ai_usage_reset();\n" + \
        f"CREATE TABLE out AS {sql_body}\n" + \
        "SELECT 'USAGE', sum(llm_calls - embed_calls), sum(input_tokens), sum(cache_write_tokens), sum(cached_tokens), round(sum(cost_usd), 4) FROM ai_usage();\n" + \
        ("SELECT 'BP', breakpoint_requests FROM ai_mock_breakpoints();\n" if MOCK else "")
    p = subprocess.run([BIN, "-csv", "-noheader"], input=script, env=ENV, text=True, capture_output=True, timeout=3600)
    usage = next((l for l in p.stdout.splitlines() if l.startswith("USAGE,")), None)
    if not usage:
        return None, (p.stderr.strip().splitlines() or ["no output"])[-1][:200]
    f = usage.split(","); bp = next((l.split(",")[1] for l in p.stdout.splitlines() if l.startswith("BP,")), "")
    return {"calls": int(f[1]), "input": int(f[2]), "written": int(f[3]), "cached": int(f[4]), "cost": float(f[5]), "bp": bp}, None

def main():
    which = [a for a in sys.argv[1:] if a.startswith("P")]
    print(f"{'pattern':8s} {'calls':>6s} {'input tok':>10s} {'written':>8s} {'cached':>8s} {'cost on':>9s} {'cost off':>9s} {'saving':>7s}  description")
    for slot, (name, desc, body) in enumerate(patterns()):
        if which and name not in which:
            continue
        off, e2 = run(body, False, slot)  # the baseline first: it writes nothing, so it cannot prime the on-run
        on, e1 = run(body, True, slot)
        if e1 or e2:
            print(f"{name:8s} ERROR {e1 or e2}"); continue
        saving = (1 - on["cost"] / off["cost"]) * 100 if off["cost"] else 0.0
        print(f"{name:8s} {on['calls']:6d} {on['input']:10d} {on['written']:8d} {on['cached']:8d} {on['cost']:9.4f} {off['cost']:9.4f} {saving:6.1f}%  {desc}" + (f"  (bp {on['bp']})" if MOCK else ""))

main()
