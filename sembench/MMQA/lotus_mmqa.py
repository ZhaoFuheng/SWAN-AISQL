#!/usr/bin/env python3
"""Run LOTUS (gpt-5-mini) on SemBench MMQA over the sf_200 data, through the global cache proxy (:4001).

Mirrors the SWAN Q*.sql mapping (same tables / prompts) so LOTUS-vs-SWAN is apples-to-apples on identical
data. Records per-subquery latency, LLM cost + tokens (via lm.stats.physical_usage), and accuracy
(precision/recall/F1 vs the SemBench ground_truth, the same metric the paper uses).

  env: AI_PROXY_URL (default http://localhost:4001), AI_MODEL (default gpt-5-mini), AI_MAX_CONCURRENCY (20).
  run (from sembench/MMQA):  <swan-env>/bin/python lotus_mmqa.py [q1 q3a q3f q4 q5 q6a q6b q6c]
"""
import os, sys
# LOTUS renders multi-column prompts in Python-set order, which is randomized per process (PYTHONHASHSEED).
# So the SAME query produces byte-different prompts across runs -> the global cache (keyed on the request
# body) misses on every rerun for multi-column ops (sem_filter over title+text, Destinations+Airlines, ...).
# Pin the seed so prompts are stable and the cache replays. Must happen BEFORE importing lotus.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)
import time, json, urllib.request
import lotus, pandas as pd

# BENCHMARK RULE: local caches must be query-scoped (Q1's cache must not serve Q2). LOTUS's
# in-process cache is process-lifetime with no query boundary, so it must stay OFF; cross-run
# reuse is the server cache's job, which replays recorded latency AND cost on every hit.
lotus.settings.configure(enable_cache=False)
from lotus.models import LM
from lotus.dtype_extensions import ImageArray

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "files/mmqa/data/sf_200")
GT = os.path.join(HERE, "ground_truth")
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

def load(name):
    return pd.read_csv(os.path.join(DATA, name + ".csv"))

def gt(name):
    return json.load(open(os.path.join(GT, name + ".json")))["ground_truth"]

def norm(s):
    return " ".join(str(s).strip().lower().split())

def prf(out, gold):
    O, G = {norm(x) for x in out}, {norm(x) for x in gold}
    tp = len(O & G)
    P = tp / len(O) if O else (1.0 if not G else 0.0)
    R = tp / len(G) if G else 1.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return round(P, 3), round(R, 3), round(F, 3)

def newcol(before, after):
    extra = [c for c in after.columns if c not in before.columns]
    return extra[0] if extra else after.columns[-1]

BASE24 = ['Orange County','Mean Girls','Love Is the Drug','Crashing','Cloverfield',"My Best Friend's Girl",
    'Crossing Over','Hot Tub Time Machine','The Last Rites of Ransom Pride','127 Hours','High Road',
    'Save the Date','Bachelorette','3, 2, 1... Frankie Go Boom','Queens of Country','Item 47','The Interview',
    'The Night Before','Now You See Me 2','Allied','The Disaster Artist','Extinction',
    'The People We Hate at the Wedding','Cobweb']
MOVIES16 = ['Love Is the Drug','Crashing','Cloverfield',"My Best Friend's Girl",'Hot Tub Time Machine',
    'The Last Rites of Ransom Pride','Save the Date','Bachelorette','3, 2, 1... Frankie Go Boom',
    'Queens of Country','Item 47','The Night Before','Now You See Me 2','Allied','Extinction','Cobweb']

# ---- queries: each returns (output_iterable, gold_iterable) ----

def q1():  # director (table+text). bigquery uses AI.GENERATE("Extract the director name...") over EVERY row
    # then merges+filters -- extract-all (~200 calls), the naive dataflow. LOTUS's AI.GENERATE equivalent is
    # sem_map. (SWAN reduces the identical structure to 1 call via DuckDB's dynamic join filter.)
    bp, txt = load("ben_piazza"), load("ben_piazza_text_data")
    ext = txt.sem_map("Extract the director name from the following movie description: {text}",
                      suffix="director")
    j = pd.merge(bp, ext, left_on="Title", right_on="title", how="left")
    return list(j[j.Role == "Bob Whitewood"]["director"]), gt("q1")

def q3(phrase, key):  # genre semantic filter over 200 movie descriptions (bigquery AI.IF wording)
    liz = load("lizzy_caplan_text_data")
    r = liz.sem_filter("{title} is a " + phrase + " given their description: {text}")
    return list(r["title"]), gt(key)

def q4():  # categorize the 24 base movies by genre (multi-label) -> (genre, movie) pairs
    liz = load("lizzy_caplan_text_data")
    sub = liz[liz.title.isin(BASE24)].copy()
    r = sub.sem_map("Extract all applicable genres for each movie based on their description (respond as a "
                    "comma-separated list of lowercase genre labels, no other text): {text}", suffix="genres")
    out = []
    for _, row in r.iterrows():
        for g in str(row["genres"]).split(","):
            g = g.strip().lower()
            if g:
                out.append((g, norm(row["title"])))
    gold = [(norm(g), norm(m)) for g, ms in gt("q4").items() for m in ms]
    return out, gold

def q5():  # actor in ALL 16 movies (text agg)
    liz = load("lizzy_caplan_text_data")
    sub = liz[liz.title.isin(MOVIES16)]
    r = sub.sem_agg("Each document is the description of a movie. Identify the single actor or actress who "
                    "has played a role in ALL of these movies. Respond with ONLY that person's name. {text}")
    col = newcol(sub, r)
    return [str(r.iloc[0][col])], gt("q5")

def q6(place, key):  # airlines with destinations in <place>
    tampa = load("tampa_international_airport")
    r = tampa.sem_filter("Given destinations '{Destinations}' of {Airlines}, the airline has flights to "
                         + place + ".")
    return list(r["Airlines"]), gt(key)

IMGDIR = os.path.join(DATA, "images")

def _images_df():  # thalamusdb_images with an ImageArray column of absolute paths
    im = load("thalamusdb_images").copy()
    im["image"] = ImageArray([os.path.join(IMGDIR, f) for f in im.image_filename])
    return im[["image_filename", "image"]]

def _q2_matches():  # (ID, Track, image_filename) rows where the image is the track's logo -- shared q2a/q2b
    w = load("ap_warrior")[["ID", "Track"]]
    cross = w.merge(_images_df(), how="cross")
    cross["image"] = ImageArray(list(cross["image"]))  # re-wrap: merge drops the extension dtype
    return cross.sem_filter("The image {image} shows the logo of the racetrack: {Track}")

def q2a():  # racetrack logos (table+image): (ID, image) pairs
    r = _q2_matches()
    out = [(int(t.ID), t.image_filename) for t in r.itertuples()]
    return out, [(int(a), b) for a, b in gt("q2a")]

def q2b():  # q2a + the logo's dominant color: (ID, image, color) triples
    r = _q2_matches()
    if len(r):
        r = r.sem_map("What is the single dominant color of the logo in this image? Answer with ONE color "
                      "word only. Image: {image}", suffix="color")
    out = [(int(t.ID), t.image_filename, norm(getattr(t, "color", ""))) for t in r.itertuples()]
    return out, [(int(a), b, norm(c)) for a, b, c in gt("q2b")]

def q7():  # for each Europe airline, its logo (table+image). Chains from the Europe filter (q6c logic).
    tampa = load("tampa_international_airport")
    europe = tampa.sem_filter("Given destinations '{Destinations}' of {Airlines}, the airline has flights "
                              "to Europe.")
    cross = europe[["Airlines"]].merge(_images_df(), how="cross")
    cross["image"] = ImageArray(list(cross["image"]))
    r = cross.sem_filter("The image {image} shows the logo of the airline: {Airlines}")
    out = [(t.Airlines, t.image_filename) for t in r.itertuples()]
    return out, [(a, b) for a, b in gt("q7")]

QUERIES = {
    "q1": q1,
    "q2a": q2a,
    "q2b": q2b,
    "q3a": lambda: q3("comedy movie", "q3a"),
    "q3f": lambda: q3("romantic comedy", "q3f"),
    "q4": q4,
    "q5": q5,
    "q6a": lambda: q6("Frankfurt", "q6a"),
    "q6b": lambda: q6("Germany", "q6b"),
    "q6c": lambda: q6("Europe", "q6c"),
    "q7": q7,
}

def main():
    from datetime import datetime, timezone
    which = sys.argv[1:] or list(QUERIES.keys())
    RES = os.path.join(HERE, "lotus_mmqa_results.json")   # merged per-query metrics (authoritative)
    RAW = os.path.join(HERE, "lotus_mmqa_raw.json")       # predicted vs gold sets, for auditing accuracy
    LOG = os.path.join(HERE, "lotus_mmqa_log.txt")        # append-only human-readable run log
    def _load(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    per_query = _load(RES).get("per_query", {})           # MERGE: keep prior queries, never clobber
    raw = _load(RAW)

    lm = LM(model=MODEL, api_base=PROXY, api_key="dummy", max_batch_size=CONC)
    lotus.settings.configure(lm=lm)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header = f"LOTUS {MODEL} via {PROXY}  concurrency={CONC}  queries={which}"
    print(header + "\n")
    lines = []
    for name in which:
        lm.reset_stats()
        m0 = cache_misses()
        r0 = cache_requests()
        t0 = time.time()
        try:
            out, gold = QUERIES[name]()
            err = None
        except Exception as e:
            out, gold, err = [], [], repr(e)[:300]
        dt = time.time() - t0
        P, R, F = prf(out, gold) if err is None else (0.0, 0.0, 0.0)
        u = lm.stats.physical_usage
        rec = {"latency_s": round(dt, 1), "llm_calls_fresh": cache_misses() - m0,
               "llm_calls": cache_requests() - r0,
               "tokens": u.total_tokens, "cost_usd": round(u.total_cost, 5),
               "n_out": len(out), "n_gold": len(gold), "precision": P, "recall": R, "f1": F,
               "error": err, "ran_at": stamp}
        per_query[name] = rec
        raw[name] = {"predicted": [list(x) if isinstance(x, tuple) else x for x in out],
                     "gold": [list(x) if isinstance(x, tuple) else x for x in gold]}
        tag = f"ERR {err}" if err else f"P={P} R={R} F1={F}"
        line = (f"{name:5} lat={rec['latency_s']:6.1f}s calls={rec['llm_calls_fresh']:4} "
                f"tok={rec['tokens']:7} ${rec['cost_usd']:.4f}  n={rec['n_out']}/{rec['n_gold']}  {tag}")
        print(line)
        lines.append(line)
    allq = per_query
    tot = {"n_queries": len(allq),
           "latency_s": round(sum(r["latency_s"] for r in allq.values()), 1),
           "cost_usd": round(sum(r["cost_usd"] for r in allq.values()), 4),
           "tokens": sum(r["tokens"] for r in allq.values()),
           "llm_calls_fresh": sum(r["llm_calls_fresh"] for r in allq.values()),
           "macro_f1": round(sum(r["f1"] for r in allq.values()) / len(allq), 3)}
    tline = (f"TOTAL({len(allq)}q)  latency={tot['latency_s']}s  cost=${tot['cost_usd']}  "
             f"tokens={tot['tokens']}  fresh_calls={tot['llm_calls_fresh']}  macro-F1={tot['macro_f1']}")
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
