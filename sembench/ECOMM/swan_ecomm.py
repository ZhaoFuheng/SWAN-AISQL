#!/usr/bin/env python3
"""SWAN AI-SQL on SemBench ECOMM sf_500 -- head-to-head companion to lotus_ecomm.py.

Same data, cache proxy (:4001), concurrency, and metrics (f1 / adjusted-rand-index per the TOMLs).
Queries are translated from the bigquery + lotus dialects with the SAME prompt text and the SAME relational
pre-filters, so the comparison is engine-vs-engine, not prompt-vs-prompt.
  AI.IF -> ai_filter    AI.GENERATE -> ai_complete    AI.CLASSIFY(categories=>) -> ai_classify(struct)
  images -> ai_image(filepath)

  run (from sembench/ECOMM):  python3 swan_ecomm.py [q1 q2 ...]
  env: AI_PROXY_URL (:4001), AI_MODEL (gpt-5-mini), AI_MAX_CONCURRENCY (20); optimizer flags at defaults.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

from ecomm_common import ARI_QUERIES, gt, score

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
DB = os.path.join(HERE, "ecomm.db")
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = os.environ.get("AI_MAX_CONCURRENCY", "20")
# SWAN_TAG suffixes the result files (e.g. "_typesafe") so a variant run never clobbers the default run.
TAG = os.environ.get("SWAN_TAG", "")
ENV = {**os.environ, "AI_PROXY_URL": PROXY, "AI_MODEL": MODEL, "AI_API_KEY": "sk-test",
       "AI_MAX_CONCURRENCY": CONC, "AI_TIMEOUT_MS": "180000"}
# Per-query temp paths so concurrent queries never collide.
def outf(name):
    return f"/tmp/swan_ecomm_out_{name}.json"


def usef(name):
    return f"/tmp/swan_ecomm_usage_{name}.json"

DESC = "coalesce(description, '')"
CLS5 = ("{'Dress': 'A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, "
        "and often consist of a bodice attached to a skirt.', "
        "'Bottomwear': 'Bottomwear refers to clothing worn on the lower part of the body, such as trousers, "
        "jeans, skirts, shorts, and leggings.', "
        "'Socks': 'Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to "
        "provide comfort and warmth.', "
        "'Topwear': 'Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, "
        "t-shirts, and jackets', "
        "'Innerwear': 'Innerwear refers to clothing worn beneath outer garments, typically close to the skin, "
        "such as underwear, bras, and undershirts.'}")

SQL = {
 "q1": f"""SELECT id FROM styles_details
   WHERE ai_filter('The product is a backpack from Reebok: ' || title || ' ' || {DESC});""",

 "q2": """SELECT i.id FROM images i
   WHERE ai_filter('The image shows a (pair of) sports shoe(s) that feature the colors yellow and silver. '
                   || ai_image(i.filepath));""",

 "q3": f"""SELECT id, ai_complete('Extract the brand name from the following product description. '
     || 'Only return the brand name, nothing else: ' || title || ' ' || {DESC}) AS category
   FROM styles_details;""",

 "q4": """SELECT s.id, ai_complete('Extract the primary color of the product in the image. '
     || 'Only return the base color, nothing else. ' || ai_image(i.filepath)) AS category
   FROM styles_details s JOIN images i ON i.id = s.id
   WHERE s.baseColour IN ('Black','Blue','Red','White','Orange','Green');""",

 "q5": f"""SELECT id, ai_classify('You are given a description of a product. Your task is to classify the '
     || 'product. The product description is as follows: ' || title || ' ' || {DESC}, {CLS5}) AS category
   FROM styles_details
   WHERE masterCategory = 'Apparel' AND subCategory NOT IN ('Saree','Apparel Set','Loungewear and Nightwear');""",

 "q6": f"""SELECT s.id, ai_classify('You are given an image of a product. Your task is to classify the '
     || 'product. The product image is as follows: ' || ai_image(i.filepath), {CLS5}) AS category
   FROM styles_details s JOIN images i ON i.id = s.id
   WHERE s.masterCategory = 'Apparel'
     AND s.subCategory NOT IN ('Saree','Apparel Set','Loungewear and Nightwear');""",

 "q7": f"""WITH ps AS (SELECT * FROM styles_details WHERE price <= 500)
   SELECT p1.id || '-' || p2.id AS id FROM ps p1, ps p2
   WHERE ai_filter('You will be given two product descriptions. Do both product descriptions describe '
     || 'products of the same category from the same brand, e.g., both are t-shirts from Adidas? '
     || 'The first product description is: ' || p1.title || ' - ' || coalesce(p1.description,'')
     || ' The second product description is: ' || p2.title || ' - ' || coalesce(p2.description,''));""",

 "q8": f"""WITH sel AS (SELECT * FROM styles_details WHERE length({DESC}) >= 3000)
   SELECT s.id || '-' || i.id AS id FROM sel s, images i
   WHERE ai_filter('The image ' || ai_image(i.filepath) || ' fits the description: '
                   || s.title || ' ' || coalesce(s.description,''));""",

 "q9": """WITH ps AS (SELECT s.id, i.filepath FROM styles_details s JOIN images i ON i.id = s.id
     WHERE s.baseColour IN ('Black','Blue','Red','White','Orange','Green')
       AND s.colour1 = '' AND s.colour2 = '' AND s.price < 800)
   SELECT p1.id || '-' || p2.id AS id FROM ps p1, ps p2
   WHERE p1.id != p2.id
     AND ai_filter('The two images depict objects of the same category and the same dominant surface color. '
       || 'The first image is ' || ai_image(p1.filepath)
       || ' The second image is ' || ai_image(p2.filepath));""",

 "q10": """WITH pre AS (SELECT s.id, s.title, coalesce(s.description,'') AS description, i.filepath
     FROM styles_details s JOIN images i ON i.id = s.id
     WHERE s.baseColour IN ('Black','Blue','Red','White') AND s.price <= 1000),
   footwear AS (SELECT * FROM pre WHERE ai_filter('The image depicts a (pair of) shoe(s), sandal(s), '
     || 'flip-flop(s). If there are multiple products in the picture, always refer to the most prominent one. '
     || ai_image(filepath))),
   bottomwear AS (SELECT * FROM pre WHERE ai_filter('The image depicts a piece of apparel that can be worn '
     || 'on the lower part of the body, like pants, shorts, skirts, ... If there are multiple products in the '
     || 'picture, always refer to the most prominent one. ' || ai_image(filepath))),
   topwear AS (SELECT * FROM pre WHERE ai_filter('The image depicts a piece of apparel that can be worn on '
     || 'the upper part of the body, like t-shirts, shirts, pullovers, hoodies, but still require some sort '
     || 'of clothing on the lower body, which means, e.g., not a dress. If there are multiple products in the '
     || 'picture, always refer to the most prominent one. ' || ai_image(filepath)))
   SELECT f.id || '-' || b.id || '-' || t.id AS id
   FROM footwear f, bottomwear b, topwear t
   WHERE ai_filter('The images depict products with the same primary base color, e.g., both are black, and '
       || 'both products are from the same brand. The description of the first product is ' || f.title || ' '
       || f.description || ' and the image of the first product is ' || ai_image(f.filepath)
       || '. The description of the second product is ' || b.title || ' ' || b.description
       || ' and the image of the second product is ' || ai_image(b.filepath))
     AND ai_filter('The images depict products with the same primary base color, e.g., both are black, and '
       || 'both products are from the same brand. The description of the first product is ' || b.title || ' '
       || b.description || ' and the image of the first product is ' || ai_image(b.filepath)
       || '. The description of the second product is ' || t.title || ' ' || t.description
       || ' and the image of the second product is ' || ai_image(t.filepath));""",

 "q11": """WITH pre AS (SELECT s.id, s.title, coalesce(s.description,'') AS description, s.price, i.filepath
     FROM styles_details s JOIN images i ON i.id = s.id),
   footwear AS (SELECT * FROM pre WHERE ai_filter('You will receive an image and a description of a product. '
     || 'Determine whether the product can be worn on the feet, like shoes, sandals, flip-flops, ... '
     || 'The predominant color of the depicted product should be black. If there are multiple products in the '
     || 'picture, always refer to the most prominent one. The description of the product is as follows: '
     || title || ' ' || description || ' ' || ai_image(filepath))),
   bottomwear AS (SELECT * FROM pre WHERE ai_filter('You will receive an image and a description of a '
     || 'product. Determine whether the product can be worn on the lower part of the body, like pants, '
     || 'shorts, skirts, ... The predominant color of the depicted product should be black. Do not consider '
     || 'swimwear. If there are multiple products in the picture, always refer to the most prominent one. '
     || 'The description of the product is as follows: ' || title || ' ' || description || ' '
     || ai_image(filepath))),
   topwear AS (SELECT * FROM pre WHERE ai_filter('You will receive an image and a description of a product. '
     || 'Determine whether the product can be worn on the upper part of the body, like t-shirts, shirts, '
     || 'pullovers, hoodies, but still require some sort of clothing on the lower body, which means, e.g., '
     || 'not a dress. The predominant color of the depicted product should be black. Do not consider '
     || 'swimwear. If there are multiple products in the picture, always refer to the most prominent one. '
     || 'The description of the product is as follows: ' || title || ' ' || description || ' '
     || ai_image(filepath))),
   accessories AS (SELECT * FROM pre WHERE price <= 500 AND ai_filter('You will receive an image and a '
     || 'description of a product. Determine whether the product a watch or some jewellery or a bag. A bag '
     || 'might be a handbag or a (gym) backpack or some other type of bag. If there are multiple products in '
     || 'the picture, always refer to the most prominent one. The description of the product is as follows: '
     || title || ' ' || description || ' ' || ai_image(filepath)))
   SELECT f.id || '-' || b.id || '-' || t.id || '-' || a.id AS id
   FROM footwear f, accessories a, bottomwear b, topwear t
   WHERE ai_filter('You will receive a description and an image of two products. Determine whether they are '
       || 'from the same brand. The description of the first product is as follows: ' || f.title || ' '
       || f.description || '. And the image of the first product is ' || ai_image(f.filepath)
       || '. The description of the second product is as follows: ' || a.title || ' ' || a.description
       || '. And the image of the second product is ' || ai_image(a.filepath))
     AND ai_filter('You will receive a description and an image of two products. Determine whether they are '
       || 'from the same brand. The description of the first product is as follows: ' || b.title || ' '
       || b.description || '. And the image of the first product is ' || ai_image(b.filepath)
       || '. The description of the second product is as follows: ' || t.title || ' ' || t.description
       || '. And the image of the second product is ' || ai_image(t.filepath))
     AND ai_filter('You will receive a description and an image of two products. Determine whether they are '
       || 'from the same brand. The description of the first product is as follows: ' || a.title || ' '
       || a.description || '. And the image of the first product is ' || ai_image(a.filepath)
       || '. The description of the second product is as follows: ' || b.title || ' ' || b.description
       || '. And the image of the second product is ' || ai_image(b.filepath));""",

 "q12": f"""SELECT '{{"id":' || s.id || ',"brand":"'
     || lower(ai_complete('Extract the brand name from the description and/or image. Only return the brand '
                          || 'name in lower-case letters, nothing else: ' || s.title || ' ' || {DESC}
                          || ' ' || ai_image(i.filepath)))
     || '","category":"'
     || lower(ai_classify('Classify the product shown in the image with the given description: ' || s.title
                          || ' ' || {DESC} || ' ' || ai_image(i.filepath),
                          ['accessories','apparel','footwear']))
     || '"}}' AS id
   FROM styles_details s JOIN images i ON i.id = s.id
   WHERE s.masterCategory IN ('Accessories','Apparel','Footwear')
     AND ai_filter('Does the following description describe a product from either Adidas or Puma? '
                   || s.title || ' ' || {DESC});""",

 "q13": f"""SELECT s.id FROM styles_details s JOIN images i ON i.id = s.id
   WHERE ai_filter('You will receive a description of what a customer is looking for together with an image '
     || 'and a textual description of the product. Determine if they both match. I am looking for a running '
     || 'shirt for men with a round neck and short sleeves, preferably in blue or black, but not bright '
     || 'colors like white. Also definitely not green. It should be suitable for outdoor running in warm '
     || 'weather. If the t-shirt is not green, it should at least feature a striped design. The product has '
     || 'the following image ' || ai_image(i.filepath) || ' and textual description ' || s.title || ' '
     || {DESC});""",

 "q14": f"""WITH sel AS (SELECT * FROM styles_details WHERE price < 130),
   wsocks AS (SELECT * FROM images
     WHERE ai_filter('The image ' || ai_image(filepath) || ' depicts white socks')),
   scored AS (SELECT s.id AS sid, w.id AS wid,
       ai_score('The image ' || ai_image(w.filepath) || ' fits the description: ' || s.title || ' '
                || coalesce(s.description,''), 'how well the image fits the description') AS sc
     FROM sel s, wsocks w
     WHERE ai_filter('The image ' || ai_image(w.filepath) || ' fits the description: ' || s.title || ' '
                     || coalesce(s.description,'')))
   SELECT wid::VARCHAR AS id FROM scored
   QUALIFY row_number() OVER (PARTITION BY sid ORDER BY sc DESC) = 1;""",
}


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
    # -readonly: the queries are SELECT-only, and read-only connections don't take the exclusive db
    # lock -- required for the concurrent (default) mode where all queries run at once.
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


def extract(name, rows):
    if name in ARI_QUERIES:
        return [(str(r["id"]), str(r["category"])) for r in rows]
    return [str(r["id"]) for r in rows]


def run_one(name, stamp):
    """Run one query end-to-end -> (record, raw_entry, printable line). Thread-safe."""
    t0 = time.time()
    rows, usage, err_txt = run_sql(name, SQL[name])
    dt = time.time() - t0
    try:
        out, gold, err = extract(name, rows), gt(name), None
    except Exception as ex:
        out, gold, err = [], [], repr(ex)[:200]
    if not rows and err_txt.strip():
        err = err or ("sql: " + err_txt.strip().splitlines()[-1][:200])
    P, R, M = score(name, out, gold) if err is None else (0.0, 0.0, 0.0)
    metric = "ari" if name in ARI_QUERIES else "f1"
    rec = {"latency_s": round(dt, 1), "llm_calls_fresh": None,
           "llm_calls": usage.get("llm_calls"), "cache_hits": usage.get("cache_hits"),
           "tokens": usage.get("total_tokens"), "reasoning_tokens": usage.get("reasoning_tokens"),
           "cost_usd": round(usage.get("cost_usd") or 0.0, 5),
           "n_out": len(out), "n_gold": len(gold), "precision": P, "recall": R,
           "metric": metric, "quality": M, "error": err, "ran_at": stamp}
    raw_entry = {"predicted": [list(x) if isinstance(x, tuple) else x for x in out][:3000],
                 "gold": [list(x) if isinstance(x, tuple) else x for x in gold][:3000]}
    tag = f"ERR {err}" if err else f"P={P} R={R} {metric}={M}"
    line = (f"{name:4} lat={rec['latency_s']:7.1f}s calls={str(rec['llm_calls']):>6} "
            f"fresh={str(rec['llm_calls_fresh']):>5} tok={str(rec['tokens']):>8} ${rec['cost_usd']:.4f}  "
            f"n={rec['n_out']}/{rec['n_gold']}  {tag}")
    return rec, raw_entry, line


def main():
    from datetime import datetime, timezone
    args = sys.argv[1:]
    # Concurrent by default: cache-replayed dev runs overlap their (simulated) call latencies, so the
    # sweep takes ~the slowest query instead of the sum. --serial restores one-at-a-time (use it for
    # a clean fresh run, or when per-query fresh-call attribution matters -- fresh is sweep-level here).
    serial = "--serial" in args
    which = [a for a in args if a != "--serial"] or [f"q{i}" for i in range(1, 15)]
    RES, RAW, LOG = (os.path.join(HERE, f) for f in
                     (f"swan_ecomm_results{TAG}.json", f"swan_ecomm_raw{TAG}.json", f"swan_ecomm_log{TAG}.txt"))
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
    header = f"SWAN ECOMM {MODEL} via {PROXY}  concurrency={CONC}  sf500  {mode}  queries={which}"
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
    print(f"logged -> {os.path.basename(RES)} / {os.path.basename(RAW)} / {os.path.basename(LOG)}")


if __name__ == "__main__":
    main()
