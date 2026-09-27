#!/usr/bin/env python3
"""LOTUS on SemBench ECOMM sf_500 -- gpt-5.6-luna through the global cache proxy (:4001), concurrency 20.

Runs the VENDORED official SemBench LOTUS dialect scripts (lotus_queries/qN.py, unmodified `run(data_dir)`
functions) and scores them with the TOML metrics: f1 over the output ids, adjusted-rand-index for q3-q6.
Reports llm calls (fresh via proxy misses), tokens+cost (litellm physical usage), wall latency, quality.

  run (from sembench/ECOMM):  <swan-env>/bin/python lotus_ecomm.py [q1 q2 ...]
"""
import os, sys
# Pin the hash seed BEFORE importing lotus: multi-column prompts render in set order -> unstable cache keys.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)
import importlib.util
import json
import time
import urllib.request

import lotus

# BENCHMARK RULE: local caches must be query-scoped (Q1's cache must not serve Q2). LOTUS's
# in-process cache is process-lifetime with no query boundary, so it must stay OFF; cross-run
# reuse is the server cache's job, which replays recorded latency AND cost on every hit.
lotus.settings.configure(enable_cache=False)
from lotus.models import LM

from ecomm_common import ARI_QUERIES, gt, score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "sf_500")
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = int(os.environ.get("AI_MAX_CONCURRENCY", "20"))


def cache_requests():
    try:
        s = json.load(urllib.request.urlopen(PROXY + "/cache/stats", timeout=5))
        return s["hits"] + s["misses"]
    except Exception:
        return 0


def cache_misses():
    try:
        return json.load(urllib.request.urlopen(PROXY + "/cache/stats", timeout=5))["misses"]
    except Exception:
        return -1


def load_query(name):
    p = os.path.join(HERE, "lotus_queries", name + ".py")
    spec = importlib.util.spec_from_file_location("lq_" + name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract(name, res):
    # ARI queries return a DataFrame [id, category]; the rest a Series/list of ids (q12: json strings).
    if name in ARI_QUERIES:
        cols = list(res.columns)
        cat = "category" if "category" in cols else cols[-1]
        return [(str(r["id"]), str(r[cat])) for _, r in res.iterrows()]
    return [str(x) for x in list(res)]


def main():
    from datetime import datetime, timezone
    which = sys.argv[1:] or [f"q{i}" for i in range(1, 15)]
    RES, RAW, LOG = (os.path.join(HERE, f) for f in
                     ("lotus_ecomm_results.json", "lotus_ecomm_raw.json", "lotus_ecomm_log.txt"))
    def _load(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    per_query = _load(RES).get("per_query", {})
    raw = _load(RAW)

    lm = LM(model=MODEL, api_base=PROXY, api_key="dummy", max_batch_size=CONC)
    lotus.settings.configure(lm=lm)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header = f"LOTUS ECOMM {MODEL} via {PROXY}  concurrency={CONC}  sf500  queries={which}"
    print(header + "\n")
    lines = []
    for name in which:
        lm.reset_stats()
        m0 = cache_misses()
        r0 = cache_requests()
        t0 = time.time()
        try:
            out = extract(name, load_query(name).run(DATA))
            gold, err = gt(name), None
        except Exception as e:
            out, gold, err = [], [], repr(e)[:300]
        dt = time.time() - t0
        P, R, M = score(name, out, gold) if err is None else (0.0, 0.0, 0.0)
        u = lm.stats.physical_usage
        metric = "ari" if name in ARI_QUERIES else "f1"
        rec = {"latency_s": round(dt, 1), "llm_calls_fresh": cache_misses() - m0,
               "llm_calls": cache_requests() - r0,
               "tokens": u.total_tokens, "cost_usd": round(u.total_cost, 5),
               "n_out": len(out), "n_gold": len(gold), "precision": P, "recall": R,
               "metric": metric, "quality": M, "error": err, "ran_at": stamp}
        per_query[name] = rec
        raw[name] = {"predicted": [list(x) if isinstance(x, tuple) else x for x in out][:3000],
                     "gold": [list(x) if isinstance(x, tuple) else x for x in gold][:3000]}
        tag = f"ERR {err}" if err else f"P={P} R={R} {metric}={M}"
        line = (f"{name:4} lat={rec['latency_s']:7.1f}s fresh={rec['llm_calls_fresh']:5} "
                f"tok={rec['tokens']:8} ${rec['cost_usd']:.4f}  n={rec['n_out']}/{rec['n_gold']}  {tag}")
        print(line, flush=True)
        lines.append(line)
    allq = per_query
    tot = {"n_queries": len(allq),
           "latency_s": round(sum(r["latency_s"] for r in allq.values()), 1),
           "cost_usd": round(sum(r["cost_usd"] for r in allq.values()), 4),
           "tokens": sum(r["tokens"] for r in allq.values()),
           "llm_calls_fresh": sum(r["llm_calls_fresh"] for r in allq.values()),
           "macro_quality": round(sum(r["quality"] for r in allq.values()) / len(allq), 3)}
    tline = (f"TOTAL({len(allq)}q)  latency={tot['latency_s']}s  cost=${tot['cost_usd']}  "
             f"tokens={tot['tokens']}  fresh={tot['llm_calls_fresh']}  macro-quality={tot['macro_quality']}")
    print("\n" + tline)
    json.dump({"model": MODEL, "concurrency": CONC, "updated": stamp, "per_query": per_query, "total": tot},
              open(RES, "w"), indent=1)
    json.dump(raw, open(RAW, "w"), indent=1)
    with open(LOG, "a") as f:
        f.write(f"\n== {stamp} ==\n{header}\n" + "\n".join(lines) + "\n" + tline + "\n")
    print(f"logged -> {os.path.basename(RES)} / {os.path.basename(RAW)} / {os.path.basename(LOG)}")


if __name__ == "__main__":
    main()
