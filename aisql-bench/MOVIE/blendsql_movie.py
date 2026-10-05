#!/usr/bin/env python3
"""BlendSQL on SemBench MOVIE sf_2000 -- the ten queries written with BlendSQL ingredients
(blendsql_queries/qN.sql: LLMMap over a base-table or CTE column), on an in-memory DuckDB holding the
reviews, through the same proxy, model, concurrency and metrics (movie_common) as the other runners.

BlendSQL's LLMMap prompt carries a built-in one-shot example; it is stripped so every system here is
zero-shot (the same rule as the SWAN 2.0 benchmark). Calls and tokens come from BlendSQL's own model counters
(true requests, streamed); cost is priced from those tokens at the SWAN 2.0 meter's gpt-5.6-luna prices.

  run:  <python with blendsql> blendsql_movie.py [q1 q2 ...]      (SWAN_bench/.venv has it)
  env:  AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna), AI_MAX_CONCURRENCY (20)
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from movie_common import gt, metric_fields, metric_tag, score

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = os.environ.get("AI_MAX_CONCURRENCY", "20")
TAG = os.environ.get("SWAN_TAG", "")
P_IN, P_OUT = 0.20, 1.20  # $ per 1M tokens, gpt-5.6-luna (SWAN_bench/src/swan_bench/meter.py)
os.environ["BLENDSQL_ASYNC_LIMIT"] = CONC

# the one-shot block of BlendSQL's "basic" LLMMap prompt (SWAN_bench/src/swan_bench/systems/blendsql.py)
_ONE_SHOT = re.compile(r" An example is shown below\.\n\n.*?\n---\n\n", re.DOTALL)


def _zero_shot(prompt):
    return _ONE_SHOT.sub("\n\n", prompt, count=1)


def make_engine():
    import pandas as pd
    from blendsql import BlendSQL
    from blendsql.db import DuckDB
    from blendsql.ingredients import LLMMap, LLMQA
    from blendsql.models import OpenAI

    class ZeroShotOpenAI(OpenAI):
        async def _format_inputs(self, extra_body, item):
            item.prompt = _zero_shot(item.prompt)
            return await super()._format_inputs(extra_body, item)

    reviews = pd.read_csv(os.path.join(HERE, "data", "sf_2000", "Reviews.csv"))
    db = DuckDB.from_pandas({"reviews": reviews})
    llm = ZeroShotOpenAI(MODEL, api_key="sk-test", base_url=PROXY + "/v1")
    return BlendSQL(db, model=llm, ingredients={LLMMap, LLMQA}), llm


def run_one(name, stamp):
    engine, llm = make_engine()  # a fresh engine per query: no temporary tables or counters carry over
    sql = open(os.path.join(HERE, "blendsql_queries", f"{name}.sql")).read()
    t0 = time.time()
    err, out, meta = None, [], None
    try:
        smoothie = engine.execute(sql)
        out = [tuple(str(v) for v in r) for r in smoothie.pl().iter_rows()]
        meta = smoothie.meta  # BlendSQL's own accounting of the query: true requests and streamed usage
    except Exception as ex:  # noqa: BLE001 - a failed query scores 0 and is reported
        err = "blendsql: " + str(ex).strip().splitlines()[-1][:200]
    dt = time.time() - t0
    calls = getattr(meta, "num_generation_calls", 0) or 0
    p_tok, c_tok = getattr(meta, "prompt_tokens", 0) or 0, getattr(meta, "completion_tokens", 0) or 0
    gold = gt(name)
    P, R, M = score(name, out, gold) if err is None else (0.0, 0.0, 0.0)
    cost = round(p_tok / 1e6 * P_IN + c_tok / 1e6 * P_OUT, 5)
    rec = {"latency_s": round(dt, 1), "llm_calls_fresh": None, "llm_calls": calls, "cache_hits": None,
           "tokens": p_tok + c_tok, "reasoning_tokens": None, "cost_usd": cost, "cost_is_estimate": True,
           "n_out": len(out), "n_gold": len(gold), **metric_fields(name, P, R),
           "quality": M, "error": err, "ran_at": stamp}
    raw_entry = {"predicted": [list(x) for x in out][:5000]}
    tag = f"ERR {err}" if err else f"{metric_tag(name, P, R)} quality={M}"
    line = (f"{name:4} lat={rec['latency_s']:7.1f}s calls={calls:>6} tok={rec['tokens']:>9} ~${cost:.4f}  "
            f"n={rec['n_out']}/{rec['n_gold']}  {tag}")
    return rec, raw_entry, line


def main():
    which = sys.argv[1:] or [f"q{i}" for i in range(1, 11)]
    RES, RAW, LOG = (os.path.join(HERE, f) for f in
                     (f"blendsql_movie_results{TAG}.json", f"blendsql_movie_raw{TAG}.json", f"blendsql_movie_log{TAG}.txt"))

    def _load(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    per_query = _load(RES).get("per_query", {})
    raw = _load(RAW)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header = f"BlendSQL MOVIE {MODEL} via {PROXY}  concurrency={CONC}  sf2000  serial  queries={which}"
    print(header + "\n")
    lines = []
    for name in which:
        rec, raw_entry, line = run_one(name, stamp)
        per_query[name], raw[name] = rec, raw_entry
        print(line, flush=True)
        lines.append(line)
    allq = per_query
    tot = {"n_queries": len(allq),
           "latency_s": round(sum(r["latency_s"] for r in allq.values()), 1),
           "cost_usd": round(sum(r["cost_usd"] for r in allq.values()), 4), "cost_is_estimate": True,
           "tokens": sum(r.get("tokens") or 0 for r in allq.values()),
           "llm_calls": sum(r.get("llm_calls") or 0 for r in allq.values()),
           "macro_quality": round(sum(r["quality"] for r in allq.values()) / len(allq), 3)}
    tline = (f"TOTAL({len(allq)}q)  latency={tot['latency_s']}s  calls={tot['llm_calls']}  ~cost=${tot['cost_usd']}  "
             f"tokens={tot['tokens']}  macro-quality={tot['macro_quality']}")
    print("\n" + tline)
    json.dump({"model": MODEL, "concurrency": CONC, "updated": stamp, "per_query": per_query, "total": tot},
              open(RES, "w"), indent=1)
    json.dump(raw, open(RAW, "w"), indent=1)
    with open(LOG, "a") as f:
        f.write(header + "\n" + "\n".join(lines) + "\n" + tline + "\n\n")


if __name__ == "__main__":
    main()
