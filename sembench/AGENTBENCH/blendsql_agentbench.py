#!/usr/bin/env python3
"""BlendSQL on the hybrid bench's 30 queries (blendsql_queries/QN.sql, the mechanical translation of the SWAN
queries by translate_to_blendsql.py), on BlendSQL's in-memory DuckDB backend holding the same dataset files,
through the same proxy, model and concurrency as the other systems. Result frames go to results/blendsql/QN.csv
and are scored by eval_agentbench.py like every other system's.

BlendSQL's LLMMap prompt carries a built-in one-shot example; it is stripped so every system is zero-shot.
Calls and tokens are BlendSQL's own accounting of the query (true requests, streamed usage); cost is priced
from those tokens at the SWAN 2.0 meter's gpt-5.6-luna prices (an estimate, like PLOP's).

  run:  <python with blendsql> blendsql_agentbench.py [Q1 Q2 ...]      (SWAN_bench/.venv has it)
  env:  AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna), AI_MAX_CONCURRENCY (20)
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = os.environ.get("AI_MAX_CONCURRENCY", "20")
P_IN, P_OUT = 0.20, 1.20  # $ per 1M tokens, gpt-5.6-luna (SWAN_bench/src/swan_bench/meter.py)
os.environ["BLENDSQL_ASYNC_LIMIT"] = CONC

_ONE_SHOT = re.compile(r" An example is shown below\.\n\n.*?\n---\n\n", re.DOTALL)


def _zero_shot(prompt):
    return _ONE_SHOT.sub("\n\n", prompt, count=1)


def make_engine(setup):
    import pandas as pd
    from blendsql import BlendSQL
    from blendsql.db import DuckDB
    from blendsql.ingredients import LLMMap, LLMQA
    from blendsql.models import OpenAI

    class ZeroShotOpenAI(OpenAI):
        async def _format_inputs(self, extra_body, item):
            item.prompt = _zero_shot(item.prompt)
            return await super()._format_inputs(extra_body, item)

    db = DuckDB.from_pandas({"__init": pd.DataFrame({"x": [0]})})
    for stmt in setup:  # the query's own CREATE TABLE / VIEW statements over ./dataset
        db.con.sql(stmt)
    llm = ZeroShotOpenAI(MODEL, api_key="sk-test", base_url=PROXY + "/v1")
    return BlendSQL(db, model=llm, ingredients={LLMMap, LLMQA})


def split_file(text):
    setup, query = text.split("-- query (BlendSQL)\n", 1)
    setup = setup.replace("-- setup (DuckDB backend)\n", "")
    stmts = [s.strip() for s in setup.split(";\n") if s.strip() and s.strip() != ";"]
    return [s.rstrip(";") for s in stmts], query


def main():
    import pandas as pd
    which = sys.argv[1:] or [f"Q{i}" for i in range(1, 31)]
    os.makedirs(os.path.join(HERE, "results", "blendsql"), exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    os.chdir(HERE)  # the setup statements read ./dataset/...
    per_query = {}
    for name in which:
        setup, query = split_file(open(os.path.join(HERE, "blendsql_queries", name + ".sql")).read())
        t0 = time.time()
        err, out, meta = None, pd.DataFrame(), None
        try:
            engine = make_engine(setup)
            smoothie = engine.execute(query)
            out = smoothie.df() if callable(smoothie.df) else smoothie.df
            meta = smoothie.meta
        except Exception as e:  # noqa: BLE001 - a failed query is reported and scores 0
            err = repr(e)[:300]
        dt = time.time() - t0
        calls = getattr(meta, "num_generation_calls", 0) or 0
        p_tok, c_tok = getattr(meta, "prompt_tokens", 0) or 0, getattr(meta, "completion_tokens", 0) or 0
        cost = round(p_tok / 1e6 * P_IN + c_tok / 1e6 * P_OUT, 5)
        rec = {"latency_s": round(dt, 1), "llm_calls": calls, "llm_calls_fresh": None, "tokens": p_tok + c_tok,
               "cost_usd": cost, "cost_is_estimate": True, "rows": len(out), "error": err, "ran_at": stamp}
        per_query[name] = rec
        out.to_csv(os.path.join(HERE, "results", "blendsql", name + ".csv"), index=False)
        print(f"{name:>4} lat={dt:7.1f}s calls={calls:>6} tok={p_tok + c_tok:>8} ~${cost:.4f} rows={len(out)}"
              + (f"  ERR {err}" if err else ""), flush=True)
        # accumulate after each query so an interrupted run keeps partials
        path = os.path.join(HERE, "results", "blendsql_agentbench_results.json")
        old = json.load(open(path)) if os.path.exists(path) else {"per_query": {}}
        old["model"], old["concurrency"], old["updated"] = MODEL, CONC, stamp
        old["per_query"].update(per_query)
        tot = old["per_query"]
        old["total"] = {"n_queries": len(tot),
                        "latency_s": round(sum(q["latency_s"] for q in tot.values()), 1),
                        "llm_calls": sum(q["llm_calls"] for q in tot.values()),
                        "cost_usd": round(sum(q["cost_usd"] for q in tot.values()), 4), "cost_is_estimate": True,
                        "tokens": sum(q["tokens"] for q in tot.values()),
                        "errors": sum(1 for q in tot.values() if q["error"])}
        json.dump(old, open(path, "w"), indent=1)
    print("\nTOTAL", json.load(open(os.path.join(HERE, "results", "blendsql_agentbench_results.json")))["total"])


main()
