# SemBench MMQA on SWAN AI-SQL

Maps the 7 SemBench MMQA question groups (https://github.com/SemBench/SemBench, `files/mmqa/query`) to
SWAN AI-SQL, over the **sf200** dataset (the 200-row scale the paper uses).

## 1. Dataset (sf200 = 200 rows, deterministic)

Generated with SemBench's own `generate_data.py` (vendored here) at `scale_factor=200`, `RANDOM_SEED=42`
— so it **exactly reproduces the paper's 200 rows**. Source is a Google Drive ZIP (`gdown` id
`1oHXq5oxIfsyoNy9aCQ0V9G3W4puqHC6i`); requires a **modern gdown** (old 3.12 fails on Drive's large-file flow).

```bash
# from swan-ai-sql/sembench/MMQA  (needs gdown>=5 and pandas on PATH)
python3 generate_data.py --working_dir . --scale_factor 200
```

Output `files/mmqa/data/sf_200/`:

| table | rows | columns | used by |
|---|---|---|---|
| `ben_piazza` | 19 | Year, Title, Role, Notes | Q1 |
| `ben_piazza_text_data` | 200 | row_id, title, url, id, text | Q1 |
| `lizzy_caplan_text_data` | 200 | row_id, title, url, text | Q3, Q4, Q5 |
| `tampa_international_airport` | 200 | row_id, Airlines, Destinations, Airport | Q6, Q7 |
| `ap_warrior` | 13 | ID, Finish, Race, Distance, Track, Condition | Q2 |
| `images/` | 200 | *.png / *.jpg | Q2, Q7 (vision) |

Each text table = its query-relevant base rows + random distractor rows (seed 42) padded to 200 — testing
the semantic operators against 200 rows (needle-in-haystack), which is the point of the scale factor.

## 2. Query mapping (Q1–Q7)

`AI.IF(pred)` → `ai_filter(pred)`; `AI.GENERATE(prompt).result` → `ai_complete(prompt)`;
genre-array `AI.GENERATE(..., output_schema=ARRAY)` → `ai_complete` (comma-separated) + `string_split`/`unnest`;
"actor in all N movies" → `ai_agg(list(text), task)`.

| file | modality | SWAN operators | notes |
|---|---|---|---|
| `Q1.sql` | table+text | `ai_complete` + join + filter | extract director; AI runs only on the qualifying row |
| `Q2.sql` | table+**image** | — | **needs vision (SWAN is text-only)** — intended query written as a comment |
| `Q3.sql` | table | `ai_filter` (×7 genres a–g) | one semantic filter per genre |
| `Q4.sql` | table | `ai_complete` + `string_split` + `unnest` + group | multi-label genre categorization |
| `Q5.sql` | text | `ai_agg(list(text), …)` | actor appearing in all 16 movies |
| `Q6.sql` | table | `ai_filter` (×3: Frankfurt/Germany/Europe) | destination-geography reasoning |
| `Q7.sql` | table+**image** | (`ai_filter` for the Europe part) | **needs vision** for the logo part — intended query in comment |

Validated structurally on the mock backend (parse, columns, joins, JSON-free genre unnest all correct).

## 3. Vision (image support) — IMPLEMENTED

SWAN now has image input via **`ai_image(path_or_url)`**: wrap it in a prompt via concatenation and the AI
functions send the image to the multimodal model. `ai_image` marks the ref with control-char sentinels;
`BuildRequestBody` turns the prompt into a multimodal `content` array (text parts + `image_url` parts; local
files → base64 data URIs, http(s) → URL). Works with `ai_filter` / `ai_complete` / `ai_classify` / `ai_score`.

Usage (Q2/Q7): `ai_filter('Is this the logo of ' || Airlines || '? ' || ai_image(image_filepath))`.

Validated against gpt-5-mini: the BA logo → `ai_complete(...)` = "British Airways"; the racetrack logo →
"Blue Santa Anita Park logo" (Santa Anita, blue — matches the Q2b ground truth); `ai_filter` discriminates
BA-vs-not correctly.

**Caching:** the persistent cache proxy (`:4001`) caches image requests exactly like text — the base64 data
URI is part of the request body it hashes, and the same file → same base64 → same key. Verified miss→hit at
4KB / 28KB / **436KB** (~579KB base64) for both `ai_complete` and the `ai_filter`+`ai_image` Q2/Q7 path (fresh
run = N misses, replay = N hits, 0 misses, results stable). So image runs replay for $0 just like text.

**Concurrency:** no special limit for vision. Measured 0 NULLs / 0 errors at `AI_MAX_CONCURRENCY` 16, 32, and
48 (16/40/60 fresh gpt-5-mini image calls through the full `:4001→:4000→OpenAI` chain; 60-at-48 finishes in the
same ~7s as 16-at-16, i.e. genuinely parallel). SWAN's AIMD controller + 6 retries absorb any transient 429.
Run image queries at the normal bench concurrency — no `AI_MAX_CONCURRENCY=1-2` workaround needed. (Earlier
image NULLs during development were not reproducible and were a transient debugging-session artifact, not a
rate limit or a proxy body-size limit.)

## 4. Running

```bash
# from swan-ai-sql/sembench/MMQA, with the AI proxy env set (AI_PROXY_URL, AI_MODEL, AI_EMBED_URL)
../../build/reldebug/duckdb -init setup.sql -c ".read Q1.sql"   # etc.
```
`setup.sql` loads the sf_200 CSVs into tables; each `Q*.sql` then runs against them.

## 5. LOTUS baseline (`lotus_mmqa.py`)

A LOTUS (v1.2.4) implementation of the same 11 MMQA subqueries over the **same sf_200 data**, for an
apples-to-apples baseline vs SWAN. Runs `gpt-5-mini` through the **same global cache proxy** (`:4001`) at
**concurrency 20**.

```bash
# swan-env = the conda env with lotus-ai installed (litellm, sentence-transformers, faiss)
<swan-env>/bin/python lotus_mmqa.py                 # all 11 subqueries
<swan-env>/bin/python lotus_mmqa.py q3a q6c q7      # a subset
```
- **Cache**: `LM(model="gpt-5-mini", api_base="http://localhost:4001", api_key="dummy", max_batch_size=20)`.
  `max_batch_size` → litellm `batch_completion(max_workers=…)`, so 20 = concurrency 20. The script **re-execs
  with `PYTHONHASHSEED=0`** — LOTUS renders multi-column prompts in Python-set order (randomized per process),
  so without a pinned seed the SAME query produces byte-different prompts each run and the global cache (keyed
  on the request body) misses on every rerun. With the seed pinned, reruns replay for $0.
- **Logged** (append/merge, never clobbered): `lotus_mmqa_results.json` (per-query metrics), `…_raw.json`
  (predicted vs gold, for auditing), `…_log.txt` (timestamped run log).
- **Accuracy** = precision/recall/F1 vs `ground_truth/*.json` (SemBench's metric).
- **Query fidelity**: SemBench ships only ONE LOTUS query for MMQA (`lotus_llama/q1.py`); q1 here follows it
  (sem_extract on every row → merge → filter). q2–q7 are original (no SemBench LOTUS exists) built from the
  bigquery logic + NL specs + the SWAN Q*.sql mapping. NB q2b is a superset of q2a (re-runs the logo filter +
  adds color), so their costs overlap — a SemBench subquery-methodology artifact, not double-spend (q2b's
  filter replays from cache; only its 17 color calls are fresh).

**Result (gpt-5-mini, sf200, 20 workers):** macro-F1 **0.76**, total standalone cost **$3.00** (~$2.03 without
the q2a/q2b filter overlap), latency dominated by the two image cross-products (q2a 616s, q7 535s). Perfect on
q1/q5/q6a/q6b (F1 1.0); genre filters q3a/q3f 0.80–0.86; weak on q4 multi-label genres (P 0.37) and the image
logo joins q2a/q2b/q7 (recall 0.8–1.0 but precision ~0.3 — many distractor images match as logos).
