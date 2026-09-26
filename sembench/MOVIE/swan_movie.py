#!/usr/bin/env python3
"""SWAN AI-SQL on SemBench MOVIE sf_2000 -- head-to-head companion to lotus_movie.py.

Same data, cache proxy (:4001), concurrency, and metrics (movie_common = faithful MovieEvaluator port).
Prompts match the official LOTUS runner's templates verbatim (rendered by SQL concat), so the comparison is
engine-vs-engine:  sem_filter -> ai_filter,  sem_join -> join + ai_filter,  sem_map classify -> ai_classify,
sem_map 1-5 scoring -> ai_score(input, criteria, 1, 5).

Concurrent by default (all queries at once, duckdb -readonly, cache-replay friendly); --serial for
one-at-a-time with per-query fresh-call attribution.

  run (from sembench/MOVIE):  python3 swan_movie.py [--serial] [q1 q2 ...]
  env: AI_PROXY_URL (:4001), AI_MODEL (gpt-5.6-luna), AI_MAX_CONCURRENCY (20); optimizer flags at defaults.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

from movie_common import gt, score

HERE = os.path.dirname(os.path.abspath(__file__))
# Provider keys come from the repo's .env (git-ignored; see .env.example): loaded here so the engine and
# the proxy see OPENAI_API_KEY / TYPESAFE_API_KEY without exporting them by hand. Existing env wins.
_ENV_FILE = os.path.join(HERE, "../../.env")
if os.path.exists(_ENV_FILE):
    for _line in open(_ENV_FILE):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
BIN = os.environ.get("DUCKDB_BIN", os.path.abspath(os.path.join(HERE, "../../build/release/duckdb")))
DB = os.path.join(HERE, "movie.db")
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = os.environ.get("AI_MAX_CONCURRENCY", "20")
# SWAN_TAG suffixes the result files (e.g. "_typesafe") so a variant run never clobbers the default run.
TAG = os.environ.get("SWAN_TAG", "")
ENV = {**os.environ, "AI_PROXY_URL": PROXY, "AI_MODEL": MODEL, "AI_API_KEY": "sk-test",
       "AI_MAX_CONCURRENCY": CONC, "AI_TIMEOUT_MS": "180000"}

POS_Q = "ai_filter('Determine if the following movie review is clearly positive. Review: \"' || reviewText || '\".')"
SAME_P = ("ai_filter('These two movie reviews express the same sentiment - either both are positive or both "
          "are negative. Review 1: \"' || r1.reviewText || '\" Review 2: \"' || r2.reviewText || '\"')")
OPP_P = ("ai_filter('These two movie reviews express opposite sentiments - one is positive and the other is "
         "negative. Review 1: \"' || r1.reviewText || '\" Review 2: \"' || r2.reviewText || '\"')")
RUBRIC = ("Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.\n\n"
          "Rubrics:\n"
          "5: Very positive. Strong positive sentiment, indicating high satisfaction.\n"
          "4: Positive. Noticeably positive sentiment, indicating general satisfaction.\n"
          "3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive "
          "without emotional language.\n"
          "2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without "
          "strong anger or frustration.\n"
          "1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or "
          "anger.\n\nReview: ")
SCORE = f"ai_score('{RUBRIC}' || reviewText, 'how much did the reviewer like the movie per the rubrics', 1, 5)"

SQL = {
 "q1": f"""SELECT reviewId FROM reviews WHERE {POS_Q} LIMIT 5;""",

 "q2": f"""SELECT reviewId FROM reviews WHERE id = 'taken_3' AND {POS_Q} LIMIT 5;""",

 "q3": """SELECT count(*) AS positive_review_cnt FROM reviews
   WHERE id = 'taken_3'
     AND ai_filter('Determine if the following review is clearly positive. Review: ' || reviewText);""",

 "q4": """SELECT sum(CASE WHEN ai_filter('Determine if the following review is clearly positive. Review: '
                                          || reviewText || '.') THEN 1 ELSE 0 END)::DOUBLE / count(*)
            AS positivity_ratio
   FROM reviews WHERE id = 'taken_3';""",

 "q5": f"""SELECT r1.id, r1.reviewId AS reviewId1, r2.reviewId AS reviewId2
   FROM reviews r1 JOIN reviews r2 ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
   WHERE r1.id = 'ant_man_and_the_wasp_quantumania' AND {SAME_P}
   LIMIT 10;""",

 "q6": f"""SELECT r1.id, r1.reviewId AS reviewId1, r2.reviewId AS reviewId2
   FROM reviews r1 JOIN reviews r2 ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
   WHERE r1.id = 'ant_man_and_the_wasp_quantumania' AND {OPP_P}
   LIMIT 10;""",

 "q7": f"""SELECT r1.id, r1.reviewId AS reviewId1, r2.reviewId AS reviewId2
   FROM reviews r1 JOIN reviews r2 ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
   WHERE r1.id = 'ant_man_and_the_wasp_quantumania' AND {OPP_P};""",

 "q8": """WITH labeled AS (
     SELECT ai_classify('Classify the sentiment of this review as either ''POSITIVE'' or ''NEGATIVE''. '
                        || 'Review: ' || reviewText, ['POSITIVE','NEGATIVE']) AS s
     FROM reviews WHERE id = 'taken_3')
   SELECT lab AS scoreSentiment, count(labeled.s) AS count
   FROM (VALUES ('NEGATIVE'), ('POSITIVE')) AS labs(lab)
   LEFT JOIN labeled ON labeled.s = labs.lab
   GROUP BY lab ORDER BY lab;""",

 "q9": f"""SELECT reviewId, {SCORE} AS reviewScore
   FROM reviews WHERE id = 'ant_man_and_the_wasp_quantumania';""",

 "q10": f"""SELECT id AS movieId, avg(score) AS movieScore
   FROM (SELECT id, {SCORE} AS score FROM reviews)
   GROUP BY id;""",
}


def outf(name):
    return f"/tmp/swan_movie_out_{name}.json"


def usef(name):
    return f"/tmp/swan_movie_usage_{name}.json"


def cache_misses():
    try:
        return json.load(urllib.request.urlopen(PROXY + "/cache/stats", timeout=5))["misses"]
    except Exception:
        return -1


def ensure_db():
    if not os.path.exists(DB):
        subprocess.run([BIN, DB], input=open(os.path.join(HERE, "setup.sql")).read(),
                       env=ENV, text=True, capture_output=True, cwd=HERE)


def run_sql(name, query):
    script = (".mode json\n.once %s\n%s\n.once %s\n"
              "SELECT (sum(llm_calls)-sum(embed_calls))::BIGINT AS llm_calls, sum(cache_hits)::BIGINT AS cache_hits, "
              "sum(total_tokens)::BIGINT AS total_tokens, sum(reasoning_tokens)::BIGINT AS reasoning_tokens, "
              "sum(cost_usd) AS cost_usd FROM ai_usage();\n" % (outf(name), query, usef(name)))
    p = subprocess.run([BIN, "-readonly", DB], input=script, env=ENV, text=True, capture_output=True, cwd=HERE)
    try:
        rows = json.load(open(outf(name)))
    except Exception:
        rows = []
    try:
        usage = json.load(open(usef(name)))[0]
    except Exception:
        usage = {}
    for f in (outf(name), usef(name)):
        try:
            os.remove(f)
        except OSError:
            pass
    return rows, usage, p.stderr


def run_one(name, stamp):
    t0 = time.time()
    rows, usage, err_txt = run_sql(name, SQL[name])
    dt = time.time() - t0
    out = [tuple(str(v) for v in r.values()) for r in rows]
    err = None
    if not rows and err_txt.strip():
        err = "sql: " + err_txt.strip().splitlines()[-1][:200]
    gold = gt(name)
    P, R, M = score(name, out, gold) if err is None else (0.0, 0.0, 0.0)
    rec = {"latency_s": round(dt, 1), "llm_calls_fresh": None,
           "llm_calls": usage.get("llm_calls"), "cache_hits": usage.get("cache_hits"),
           "tokens": usage.get("total_tokens"), "reasoning_tokens": usage.get("reasoning_tokens"),
           "cost_usd": round(usage.get("cost_usd") or 0.0, 5),
           "n_out": len(out), "n_gold": len(gold), "precision": P, "recall": R,
           "quality": M, "error": err, "ran_at": stamp}
    raw_entry = {"predicted": [list(x) for x in out][:5000]}
    tag = f"ERR {err}" if err else f"P/a={P} R/b={R} quality={M}"
    line = (f"{name:4} lat={rec['latency_s']:7.1f}s calls={str(rec['llm_calls']):>6} "
            f"fresh={str(rec['llm_calls_fresh']):>5} tok={str(rec['tokens']):>9} ${rec['cost_usd']:.4f}  "
            f"n={rec['n_out']}/{rec['n_gold']}  {tag}")
    return rec, raw_entry, line


def main():
    from datetime import datetime, timezone
    args = sys.argv[1:]
    serial = "--serial" in args
    which = [a for a in args if a != "--serial"] or [f"q{i}" for i in range(1, 11)]
    RES, RAW, LOG = (os.path.join(HERE, f) for f in
                     (f"swan_movie_results{TAG}.json", f"swan_movie_raw{TAG}.json", f"swan_movie_log{TAG}.txt"))
    def _load(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    per_query = _load(RES).get("per_query", {})
    raw = _load(RAW)
    ensure_db()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    mode = "serial" if serial else f"concurrent x{len(which)}"
    header = f"SWAN MOVIE {MODEL} via {PROXY}  concurrency={CONC}  sf2000  {mode}  queries={which}"
    print(header + "\n")
    lines = []
    m0 = cache_misses()
    wall0 = time.time()
    if serial:
        for name in which:
            f0 = cache_misses()
            rec, raw_entry, line = run_one(name, stamp)
            rec["llm_calls_fresh"] = cache_misses() - f0
            line = line.replace("fresh= None", f"fresh={rec['llm_calls_fresh']:5}")
            per_query[name], raw[name] = rec, raw_entry
            print(line, flush=True)
            lines.append(line)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=len(which)) as pool:
            futs = {pool.submit(run_one, name, stamp): name for name in which}
            for fut in as_completed(futs):
                name = futs[fut]
                rec, raw_entry, line = fut.result()
                per_query[name], raw[name] = rec, raw_entry
                print(line, flush=True)
                lines.append(line)
    wall = round(time.time() - wall0, 1)
    sweep_fresh = cache_misses() - m0
    allq = per_query
    tot = {"n_queries": len(allq),
           "latency_s": round(sum(r["latency_s"] for r in allq.values()), 1),
           "sweep_wall_s": wall,
           "cost_usd": round(sum(r["cost_usd"] for r in allq.values()), 4),
           "tokens": sum(r.get("tokens") or 0 for r in allq.values()),
           "llm_calls_fresh": sweep_fresh,
           "macro_quality": round(sum(r["quality"] for r in allq.values()) / len(allq), 3)}
    tline = (f"TOTAL({len(allq)}q)  latency={tot['latency_s']}s  wall={wall}s  cost=${tot['cost_usd']}  "
             f"tokens={tot['tokens']}  fresh={sweep_fresh}  macro-quality={tot['macro_quality']}")
    print("\n" + tline)
    json.dump({"model": MODEL, "concurrency": CONC, "updated": stamp, "per_query": per_query, "total": tot},
              open(RES, "w"), indent=1)
    json.dump(raw, open(RAW, "w"), indent=1)
    with open(LOG, "a") as f:
        f.write(f"\n== {stamp} ==\n{header}\n" + "\n".join(lines) + "\n" + tline + "\n")
    print(f"logged -> {os.path.basename(RES)}")


if __name__ == "__main__":
    main()
