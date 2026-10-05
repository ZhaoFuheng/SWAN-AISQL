#!/usr/bin/env python3
"""Run SWAN AI-SQL (gpt-5-mini) on SemBench MMQA over sf_200, through the global cache proxy (:4001).

Head-to-head companion to lotus_mmqa.py: the SAME 11 subqueries, data, cache, concurrency, and P/R/F1
accuracy metric -- so SWAN vs LOTUS is directly comparable. SWAN's own ai_usage() gives llm_calls / tokens /
cost_usd per query; latency = wall-clock; accuracy = precision/recall/F1 vs ground_truth/*.json.

  run (from aisql-bench/MMQA):  python3 swan_mmqa.py [q1 q3a q3f q4 q5 q6a q6b q6c q2a q2b q7]
  env: AI_PROXY_URL (:4001), AI_MODEL (gpt-5-mini), AI_MAX_CONCURRENCY (20). Optimizer flags off by default
       (MMQA queries are single-predicate); set DUCKDB_AI_DEDUP=1 etc. to exercise them.
"""
import os, sys, time, json, subprocess, urllib.request

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
_BUILDS = [os.path.abspath(os.path.join(HERE, "../../build", b, "duckdb")) for b in ("release", "reldebug")]
BIN = os.environ.get("DUCKDB_BIN", next((p for p in _BUILDS if os.path.exists(p)), _BUILDS[0]))
DB = os.path.join(HERE, "mmqa.db")
GT = os.path.join(HERE, "ground_truth")
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = os.environ.get("AI_MAX_CONCURRENCY", "20")
# SWAN_TAG suffixes the result files (e.g. "_typesafe") so a variant run never clobbers the default run.
TAG = os.environ.get("SWAN_TAG", "")
ENV = {**os.environ, "AI_PROXY_URL": PROXY, "AI_MODEL": MODEL, "AI_API_KEY": "sk-test",
       "AI_MAX_CONCURRENCY": CONC, "AI_TIMEOUT_MS": "180000"}
OUTF, USEF = "/tmp/swan_out.json", "/tmp/swan_usage.json"

def cache_misses():
    try:
        return json.load(urllib.request.urlopen(PROXY + "/cache/stats", timeout=5))["misses"]
    except Exception:
        return -1

def gt(name):
    return json.load(open(os.path.join(GT, name + ".json")))["ground_truth"]

def norm(s):
    return " ".join(str(s).strip().lower().split())

def prf(out, gold):
    O, G = {norm(x) if not isinstance(x, tuple) else tuple(norm(v) for v in x) for x in out}, \
           {norm(x) if not isinstance(x, tuple) else tuple(norm(v) for v in x) for x in gold}
    tp = len(O & G)
    P = tp / len(O) if O else (1.0 if not G else 0.0)
    R = tp / len(G) if G else 1.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return round(P, 3), round(R, 3), round(F, 3)

def ensure_db():
    if not os.path.exists(DB):
        subprocess.run([BIN, DB], input=open(os.path.join(HERE, "setup.sql")).read(),
                       env=ENV, text=True, capture_output=True, cwd=HERE)

def run_sql(query):
    script = (".mode json\n.once %s\n%s\n.once %s\n"
              "SELECT (sum(llm_calls)-sum(embed_calls))::BIGINT AS llm_calls, sum(cache_hits)::BIGINT AS cache_hits, "
              "sum(input_tokens)::BIGINT AS input_tokens, sum(output_tokens)::BIGINT AS output_tokens, "
              "sum(reasoning_tokens)::BIGINT AS reasoning_tokens, sum(total_tokens)::BIGINT AS total_tokens, "
              "sum(cost_usd) AS cost_usd FROM ai_usage();\n" % (OUTF, query, USEF))
    p = subprocess.run([BIN, DB], input=script, env=ENV, text=True, capture_output=True, cwd=HERE)
    try:
        rows = json.load(open(OUTF))
    except Exception:
        rows = []
    try:
        usage = json.load(open(USEF))[0]
    except Exception:
        usage = {}
    return rows, usage, p.stderr

# ---- 11 subqueries: bigquery-faithful prompts (AI.IF -> ai_filter, AI.GENERATE -> ai_complete), copied
# as-close-as-possible from files/mmqa/query/bigquery/*.sql so the comparison is on the ENGINES, not the
# prompt wording. Unavoidable deviations: (q4) SWAN has no ARRAY output_schema -> a comma-separated
# instruction stands in; (q7) bigquery is all-135-airlines x 200 images = 27k vision calls at sf200
# (impractical) -> kept Europe-chained; q5 has no bigquery reference. q1 uses bigquery's extract-all CTE.
Q3A = "SELECT title FROM lizzy_caplan_text_data WHERE ai_filter(title || ' is a comedy movie given their description: ' || text);"
Q3F = "SELECT title FROM lizzy_caplan_text_data WHERE ai_filter(title || ' is a romantic comedy given their description: ' || text);"
# Grounded (2026-07-22): the flat inline prompt let the model answer from world knowledge about the airline
# (recall 1.0, precision ~0.03); pinning the judgment to the LISTED destinations matches LOTUS's grounded
# Context/Claim behavior (measured F1 1.0 on q6a/q6b).
Q6 = ("SELECT Airlines FROM tampa_international_airport WHERE ai_filter('Given destinations ''' || "
      "Destinations || ''' of ' || Airlines || ', the airline has flights to %s. Base your answer ONLY on "
      "the listed destinations, not on outside knowledge about the airline.');")
Q1 = ("WITH movie_with_director AS (SELECT title, ai_complete('Extract the director name from the following "
      "movie description: ' || text) AS director FROM ben_piazza_text_data) "
      "SELECT t2.director FROM ben_piazza t1 JOIN movie_with_director t2 ON t1.Title = t2.title "
      "WHERE t1.Role = 'Bob Whitewood';")
Q5 = ("SELECT ai_agg(list(text), 'Each item is the description of a movie. Identify the single "
      "actor/actress who has played a role in ALL of these movies. Respond with ONLY that person''s name.') "
      "AS actor FROM lizzy_caplan_text_data WHERE title IN ('Love Is the Drug','Crashing','Cloverfield',"
      "'My Best Friend''s Girl','Hot Tub Time Machine','The Last Rites of Ransom Pride','Save the Date',"
      "'Bachelorette','3, 2, 1... Frankie Go Boom','Queens of Country','Item 47','The Night Before',"
      "'Now You See Me 2','Allied','Extinction','Cobweb');")
Q4 = ("WITH movie_genres AS (SELECT title, ai_complete('Extract all applicable genres for each movie based "
      "on their description (respond as a comma-separated list of lowercase genre labels, no other text): ' "
      "|| text) AS genres_csv "
      "FROM lizzy_caplan_text_data WHERE title IN ('Orange County','Mean Girls','Love Is the Drug',"
      "'Crashing','Cloverfield','My Best Friend''s Girl','Crossing Over','Hot Tub Time Machine',"
      "'The Last Rites of Ransom Pride','127 Hours','High Road','Save the Date','Bachelorette',"
      "'3, 2, 1... Frankie Go Boom','Queens of Country','Item 47','The Interview','The Night Before',"
      "'Now You See Me 2','Allied','The Disaster Artist','Extinction','The People We Hate at the Wedding',"
      "'Cobweb')) SELECT lower(trim(genre)) AS genre, title FROM "
      "(SELECT title, unnest(string_split(genres_csv, ',')) AS genre FROM movie_genres) WHERE trim(genre) <> '';")
Q2A = ("SELECT DISTINCT w.ID AS id, i.image_filename FROM ap_warrior w, images i WHERE ai_filter("
       "'You will be provided with a horse racetrack name and an image. Determine if the image shows the "
       "logo of the racetrack. Racetrack: ' || w.Track || ', Image: ' || ai_image(i.image_filepath)) "
       "ORDER BY id;")
Q2B = ("SELECT DISTINCT w.ID AS id, i.image_filename, ai_complete('What''s the color of the logo in the "
       "image if available: ' || ai_image(i.image_filepath) || ' Only respond with the color name.') AS color "
       "FROM ap_warrior w, images i WHERE ai_filter('You will be provided with a horse racetrack name and "
       "an image. Determine if the image shows the logo of the racetrack. Racetrack: ' || w.Track || "
       "', Image: ' || ai_image(i.image_filepath)) ORDER BY id;")
Q7 = ("WITH europe_airlines AS (SELECT Airlines FROM tampa_international_airport WHERE ai_filter("
      "'Given destinations ''' || Destinations || ''' of ' || Airlines || ', the airline has flights to "
      "Europe.')) SELECT e.Airlines, i.image_filename FROM europe_airlines e, images i WHERE ai_filter("
      "'You will be provided with an airline name and an image. Determine if the image shows the logo of "
      "the airline. Airline: ' || e.Airlines || ', Image: ' || ai_image(i.image_filepath)) "
      "ORDER BY e.Airlines;")

def genre_pairs(rows):  # q4: (genre, title) rows -> pairs; gold from the dict
    return [(r["genre"], r["title"]) for r in rows]

QUERIES = {
    "q1":  (Q1,          lambda rows: [r["director"] for r in rows],                       lambda: gt("q1")),
    "q2a": (Q2A,         lambda rows: [(r["id"], r["image_filename"]) for r in rows],       lambda: [(a, b) for a, b in gt("q2a")]),
    "q2b": (Q2B,         lambda rows: [(r["id"], r["image_filename"], r["color"]) for r in rows], lambda: [(a, b, c) for a, b, c in gt("q2b")]),
    "q3a": (Q3A, lambda rows: [r["title"] for r in rows],  lambda: gt("q3a")),
    "q3f": (Q3F, lambda rows: [r["title"] for r in rows],  lambda: gt("q3f")),
    "q4":  (Q4,          genre_pairs,                                          lambda: [(g, m) for g, ms in gt("q4").items() for m in ms]),
    "q5":  (Q5,          lambda rows: [r["actor"] for r in rows],             lambda: gt("q5")),
    "q6a": (Q6 % "Frankfurt", lambda rows: [r["Airlines"] for r in rows],     lambda: gt("q6a")),
    "q6b": (Q6 % "Germany",   lambda rows: [r["Airlines"] for r in rows],     lambda: gt("q6b")),
    "q6c": (Q6 % "Europe",    lambda rows: [r["Airlines"] for r in rows],     lambda: gt("q6c")),
    "q7":  (Q7,          lambda rows: [(r["Airlines"], r["image_filename"]) for r in rows], lambda: [(a, b) for a, b in gt("q7")]),
}

def main():
    from datetime import datetime, timezone
    which = sys.argv[1:] or list(QUERIES.keys())
    RES, RAW, LOG = (os.path.join(HERE, f) for f in
                     (f"swan_mmqa_results{TAG}.json", f"swan_mmqa_raw{TAG}.json", f"swan_mmqa_log{TAG}.txt"))
    def _load(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    per_query = _load(RES).get("per_query", {})
    raw = _load(RAW)
    ensure_db()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header = f"SWAN {MODEL} via {PROXY}  concurrency={CONC}  queries={which}"
    print(header + "\n")
    lines = []
    for name in which:
        sql, extract, gold_fn = QUERIES[name]
        m0 = cache_misses()
        t0 = time.time()
        rows, usage, err = run_sql(sql)
        dt = time.time() - t0
        try:
            out, gold, e = extract(rows), gold_fn(), None
        except Exception as ex:
            out, gold, e = [], [], repr(ex)[:200]
        if not rows and err.strip():
            e = e or ("sql: " + err.strip().splitlines()[-1][:200])
        P, R, F = prf(out, gold) if e is None else (0.0, 0.0, 0.0)
        rec = {"latency_s": round(dt, 1), "llm_calls_fresh": cache_misses() - m0,
               "llm_calls": usage.get("llm_calls"), "cache_hits": usage.get("cache_hits"),
               "tokens": usage.get("total_tokens"), "reasoning_tokens": usage.get("reasoning_tokens"),
               "cost_usd": round(usage.get("cost_usd") or 0.0, 5),
               "n_out": len(out), "n_gold": len(gold), "precision": P, "recall": R, "f1": F,
               "error": e, "ran_at": stamp}
        per_query[name] = rec
        raw[name] = {"predicted": [list(x) if isinstance(x, tuple) else x for x in out],
                     "gold": [list(x) if isinstance(x, tuple) else x for x in gold]}
        tag = f"ERR {e}" if e else f"P={P} R={R} F1={F}"
        line = (f"{name:5} lat={rec['latency_s']:6.1f}s calls={str(rec['llm_calls']):>5} "
                f"fresh={rec['llm_calls_fresh']:4} tok={str(rec['tokens']):>7} ${rec['cost_usd']:.4f}  "
                f"n={rec['n_out']}/{rec['n_gold']}  {tag}")
        print(line)
        lines.append(line)
    allq = per_query
    tot = {"n_queries": len(allq),
           "latency_s": round(sum(r["latency_s"] for r in allq.values()), 1),
           "cost_usd": round(sum(r["cost_usd"] for r in allq.values()), 4),
           "tokens": sum((r["tokens"] or 0) for r in allq.values()),
           "macro_f1": round(sum(r["f1"] for r in allq.values()) / len(allq), 3)}
    tline = (f"TOTAL({len(allq)}q)  latency={tot['latency_s']}s  cost=${tot['cost_usd']}  "
             f"tokens={tot['tokens']}  macro-F1={tot['macro_f1']}")
    print("\n" + tline)
    json.dump({"model": MODEL, "concurrency": CONC, "updated": stamp, "per_query": per_query, "total": tot},
              open(RES, "w"), indent=2)
    json.dump(raw, open(RAW, "w"), indent=2)
    with open(LOG, "a") as f:
        f.write(f"\n=== {stamp}  {header} ===\n" + "\n".join(lines) + "\n" + tline + "\n")
    print(f"\nlogged -> {os.path.basename(RES)} / {os.path.basename(RAW)} / {os.path.basename(LOG)} "
          f"({len(per_query)} queries accumulated)")

if __name__ == "__main__":
    main()
