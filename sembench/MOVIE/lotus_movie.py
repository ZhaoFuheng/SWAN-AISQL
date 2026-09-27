#!/usr/bin/env python3
"""LOTUS on SemBench MOVIE sf_2000 -- gpt-5.6-luna through the cache proxy (:4001), concurrency 20.

Query implementations are VERBATIM from SemBench's official movie lotus_runner.py (vendored at
queries/lotus_runner_reference.py): same sem_filter/sem_join/sem_map prompt templates, same head()
limits, q9/q10 use the sem_map scoring variant (the runner's ranking='map' path, matching the
published LOTUS metrics' token counts). Scored with movie_common (faithful MovieEvaluator port).

  run (from sembench/MOVIE):  <swan-env>/bin/python lotus_movie.py [q1 q2 ...]
"""
import os, sys
# Pin the hash seed BEFORE importing lotus: prompts render in set order -> unstable cache keys.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)
import json
import time
import urllib.request

import pandas as pd
import lotus

# BENCHMARK RULE: local caches must be query-scoped (Q1's cache must not serve Q2). LOTUS's
# in-process cache is process-lifetime with no query boundary, so it must stay OFF; cross-run
# reuse is the server cache's job, which replays recorded latency AND cost on every hit.
lotus.settings.configure(enable_cache=False)
from lotus.models import LM

from movie_common import gt, metric_fields, metric_tag, score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "sf_2000")
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = int(os.environ.get("AI_MAX_CONCURRENCY", "20"))

SCORING_PROMPT = """Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review: {reviewText}

Only provide the score number (1-5) with no other comments."""


def reviews_df():
    return pd.read_csv(os.path.join(DATA, "Reviews.csv"))


def q1():
    r = reviews_df().sem_filter('Determine if the following movie review is clearly positive. Review: "{reviewText}".')
    return [(str(x),) for x in r.head(5)["reviewId"]]


def q2():
    r = reviews_df()
    r = r[r["id"] == "taken_3"]
    r = r.sem_filter('Determine if the following movie review is clearly positive. Review: "{reviewText}".')
    return [(str(x),) for x in r.head(5)["reviewId"]]


def q3():
    r = reviews_df()
    r = r[r["id"] == "taken_3"]
    pos = r.sem_filter("Determine if the following review is clearly positive. Review: {reviewText}")
    return [(str(pos.shape[0]),)]


def q4():
    r = reviews_df()
    r = r[r["id"] == "taken_3"]
    pos = r.sem_filter("Determine if the following review is clearly positive. Review: {reviewText}.")
    return [(str(len(pos) / len(r)),)]


def _pair_join(instruction):
    r = reviews_df()
    r = r[r["id"] == "ant_man_and_the_wasp_quantumania"]
    joined = r.sem_join(r, join_instruction=instruction)
    joined = joined[joined["reviewId:left"] != joined["reviewId:right"]]
    return joined


SAME_INSTR = ('These two movie reviews express the same sentiment - either both are positive or both are '
              'negative. Review 1: "{reviewText:left}" Review 2: "{reviewText:right}"')
OPP_INSTR = ('These two movie reviews express opposite sentiments - one is positive and the other is '
             'negative. Review 1: "{reviewText:left}" Review 2: "{reviewText:right}"')


def q5():
    j = _pair_join(SAME_INSTR).head(10)
    return [(str(a), str(b), str(c)) for a, b, c in zip(j["id:left"], j["reviewId:left"], j["reviewId:right"])]


def q6():
    j = _pair_join(OPP_INSTR).head(10)
    return [(str(a), str(b), str(c)) for a, b, c in zip(j["id:left"], j["reviewId:left"], j["reviewId:right"])]


def q7():
    j = _pair_join(OPP_INSTR)
    return [(str(a), str(b), str(c)) for a, b, c in zip(j["id:left"], j["reviewId:left"], j["reviewId:right"])]


def q8():
    r = reviews_df()
    r = r[r["id"] == "taken_3"]
    m = r.sem_map("Classify the sentiment of this review as either 'POSITIVE' or 'NEGATIVE'. "
                  "Only output the exact word 'POSITIVE' or 'NEGATIVE' with no additional text. "
                  "Review: {reviewText}")
    counts = m["_map"].str.strip().str.upper().value_counts()
    out = []
    for lab in ("NEGATIVE", "POSITIVE"):
        out.append((lab, str(int(counts.get(lab, 0)))))
    return out


def _map_scores(df):
    scored = df.sem_map(SCORING_PROMPT)
    vals = []
    for _, row in scored.iterrows():
        try:
            v = float(str(row["_map"]).strip())
            vals.append(v if 1 <= v <= 5 else 3.0)
        except (ValueError, TypeError):
            vals.append(3.0)
    scored = scored.copy()
    scored["reviewScore"] = vals
    return scored


def q9():
    r = reviews_df()
    r = r[r["id"] == "ant_man_and_the_wasp_quantumania"]
    s = _map_scores(r)
    return [(str(a), str(b)) for a, b in zip(s["reviewId"], s["reviewScore"])]


def q10():
    s = _map_scores(reviews_df())
    out = []
    for movie_id, grp in s.groupby("id"):
        out.append((str(movie_id), str(round(grp["reviewScore"].mean(), 4))))
    return out


QUERIES = {f"q{i}": fn for i, fn in enumerate([q1, q2, q3, q4, q5, q6, q7, q8, q9, q10], start=1)}


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
    from datetime import datetime, timezone
    which = sys.argv[1:] or [f"q{i}" for i in range(1, 11)]
    RES, RAW, LOG = (os.path.join(HERE, f) for f in
                     ("lotus_movie_results.json", "lotus_movie_raw.json", "lotus_movie_log.txt"))
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
    header = f"LOTUS MOVIE {MODEL} via {PROXY}  concurrency={CONC}  sf2000  queries={which}"
    print(header + "\n")
    lines = []
    for name in which:
        lm.reset_stats()
        m0 = cache_misses()
        r0 = cache_requests()
        t0 = time.time()
        try:
            out, err = QUERIES[name](), None
        except Exception as e:
            out, err = [], repr(e)[:300]
        dt = time.time() - t0
        gold = gt(name)
        P, R, M = score(name, out, gold) if err is None else (0.0, 0.0, 0.0)
        u = lm.stats.physical_usage
        rec = {"latency_s": round(dt, 1), "llm_calls_fresh": cache_misses() - m0,
               "llm_calls": cache_requests() - r0,
               "tokens": u.total_tokens, "cost_usd": round(u.total_cost, 5),
               "n_out": len(out), "n_gold": len(gold), **metric_fields(name, P, R),
               "quality": M, "error": err, "ran_at": stamp}
        per_query[name] = rec
        raw[name] = {"predicted": [list(x) for x in out][:5000]}
        tag = f"ERR {err}" if err else f"{metric_tag(name, P, R)} quality={M}"
        line = (f"{name:4} lat={rec['latency_s']:7.1f}s calls={rec.get('llm_calls', 0):6} fresh={rec['llm_calls_fresh']:6} "
                f"tok={rec['tokens']:9} ${rec['cost_usd']:.4f}  n={rec['n_out']}/{rec['n_gold']}  {tag}")
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
    print(f"logged -> {os.path.basename(RES)}")


if __name__ == "__main__":
    main()
