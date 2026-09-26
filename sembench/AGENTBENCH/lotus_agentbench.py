#!/usr/bin/env python3
"""LOTUS baseline on the PLOP agent_bench 30 queries (translations in lotus_queries/QN.py).

Protocol matches the other suites: unmodified pip lotus-ai, in-process cache OFF (query-level
caching is the server cache's job), routed via the :4001 recording proxy, true request counting
= proxy hits+misses delta per query. No ground truth exists for this benchmark; result frames
are saved to results/lotus/QN.csv for cross-system agreement checks.

  run:  python3 lotus_agentbench.py [Q1 Q2 ...]
  env:  AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna), AI_MAX_CONCURRENCY (20)
"""
import importlib.util
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

import pandas as pd
import lotus

lotus.settings.configure(enable_cache=False)
from lotus.models import LM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = int(os.environ.get("AI_MAX_CONCURRENCY", "20"))

CSV = {
    "books_info": "book_review/books_info.csv", "reviews": "book_review/reviews.csv",
    "decades": "book_review/decades.csv", "review_context": "book_review/review_context.csv",
    "gl_business": "googlelocal/business_description.csv", "gl_review": "googlelocal/review.csv",
    "yelp_review": "yelp/review.csv", "yelp_tip": "yelp/tip.csv", "yelp_user": "yelp/user.csv",
    "yelp_business": "yelp/yelp_business_csv/business.csv",
}
PARQUET = {t: f"tpch/{t}.parquet" for t in
           ["part", "supplier", "customer", "lineitem", "orders", "nation", "region", "partsupp"]}
_frames = {}


def load(name):
    if name not in _frames:
        if name in CSV:
            _frames[name] = pd.read_csv(os.path.join(DATA, CSV[name]))
        else:
            _frames[name] = pd.read_parquet(os.path.join(DATA, PARQUET[name]))
    return _frames[name].copy()


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


def main():
    which = sys.argv[1:] or [f"Q{i}" for i in range(1, 31)]
    lm = LM(model=MODEL, api_base=PROXY + "/v1", api_key="sk-test", max_batch_size=CONC)
    lotus.settings.configure(lm=lm, enable_cache=False)
    os.makedirs(os.path.join(HERE, "results", "lotus"), exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    per_query = {}
    for name in which:
        spec = importlib.util.spec_from_file_location("lq_" + name,
                                                      os.path.join(HERE, "lotus_queries", name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        lm.reset_stats()
        r0, m0 = cache_requests(), cache_misses()
        t0 = time.time()
        err = None
        try:
            out = mod.run(load, lm)
        except Exception as e:
            out, err = pd.DataFrame(), repr(e)[:300]
        dt = time.time() - t0
        u = lm.stats.physical_usage
        rec = {"latency_s": round(dt, 1), "llm_calls": cache_requests() - r0,
               "llm_calls_fresh": cache_misses() - m0, "tokens": u.total_tokens,
               "cost_usd": round(u.total_cost, 5), "rows": len(out), "error": err, "ran_at": stamp}
        per_query[name] = rec
        out.to_csv(os.path.join(HERE, "results", "lotus", name + ".csv"), index=False)
        print(f"{name:>4} lat={dt:7.1f}s calls={rec['llm_calls']:>6} fresh={rec['llm_calls_fresh']:>6} "
              f"tok={u.total_tokens:>8} ${u.total_cost:.4f} rows={len(out)}"
              + (f"  ERR {err}" if err else ""), flush=True)
        # accumulate after each query so an interrupted run keeps partials
        path = os.path.join(HERE, "results", "lotus_agentbench_results.json")
        old = json.load(open(path)) if os.path.exists(path) else {"per_query": {}}
        old["model"], old["concurrency"], old["updated"] = MODEL, CONC, stamp
        old["per_query"].update(per_query)
        tot = old["per_query"]
        old["total"] = {"n_queries": len(tot),
                        "latency_s": round(sum(q["latency_s"] for q in tot.values()), 1),
                        "llm_calls": sum(q["llm_calls"] for q in tot.values()),
                        "cost_usd": round(sum(q["cost_usd"] for q in tot.values()), 4),
                        "tokens": sum(q["tokens"] for q in tot.values()),
                        "errors": sum(1 for q in tot.values() if q["error"])}
        json.dump(old, open(path, "w"), indent=1)
    print("\nTOTAL", json.load(open(os.path.join(HERE, "results", "lotus_agentbench_results.json")))["total"])


main()
