# SWAN AI-SQL — semantic SQL for DuckDB

`ai_filter / ai_classify / ai_score / ai_complete / ai_agg / ai_embed / ai_image` for DuckDB, plus an
optimizer and executor built around **factorized ("two-currency") execution**: every LLM call runs once per
*distinct* input, joins over AI predicates evaluate the distinct *pair domain* without materializing the
cross product, and learned predicate ordering, Yannakakis semi-join reduction and LIMIT early-stop minimize
calls. Benchmarked against LOTUS on SemBench (MOVIE / ECOMM / MMQA) and against PLOP on the hybrid bench
(agent_bench Q1–Q30). The design is in [docs/DESIGN.md](docs/DESIGN.md).

This repository is a DuckDB **extension** built on the
[extension template](https://github.com/duckdb/extension-template): the engine is the `aisql` extension,
DuckDB is the `duckdb/` submodule, CI comes from `extension-ci-tools/`, and dependencies are declared in
`vcpkg.json` (none required today).

The numbered sections are a path: build, keys, Python environment, start the serving stack, run a query,
then run SemBench MOVIE for SWAN and LOTUS and compare them. Reference material (settings, the hybrid
bench, layout) follows.

## 1. Build

```sh
git clone --recurse-submodules https://github.com/ZhaoFuheng/SWAN-AISQL
cd SWAN-AISQL
GEN=ninja make reldebug        # or: make release / make debug   (~25 min, builds DuckDB too)
```

- `build/reldebug/duckdb` — the CLI with `aisql` (plus `parquet`, `json`, `tpch`) linked in
- `build/reldebug/extension/aisql/aisql.duckdb_extension` — the loadable extension
- `build/reldebug/test/unittest` — the test runner with the extension's sqllogictests registered

The submodules track DuckDB `main`, pinned to the commit the code is written against (the engine uses
current planner/vector APIs that are not in the v1.5 release line). `vcpkg` is optional: the Makefile only
uses it when `VCPKG_TOOLCHAIN_PATH` is set, and `vcpkg.json` lists no dependencies (the LLM client uses
DuckDB's bundled http-only `httplib` and `yyjson`; TLS towards providers is terminated by the local proxies
in `serve/`).

To check the build, run the test suite. It uses an in-process deterministic mock LLM, so it needs no
network or keys:

```sh
make test_reldebug                                      # every sqllogictest under test/sql/ (python3 >= 3.10 on PATH)
build/reldebug/test/unittest "*test/sql/region/*"       # one group
```

## 2. Keys

Copy `.env.example` to `.env` (git-ignored) and fill in what you use:

```
OPENAI_API_KEY=sk-...       # required: the chat model (default gpt-5.6-luna) via litellm
TYPESAFE_API_KEY=...        # only for SET ai_typesafe='filter,classify' (ai_filter/ai_classify on TypeSafe Jev)
```

`serve/start_stack.sh` and the benchmark harnesses read `.env`; nothing else in the repo carries a key.

## 3. Python environment

The serving stack and the benchmark harnesses are Python; `.python-version` pins **3.12** (any ≥ 3.11
works). With [uv](https://docs.astral.sh/uv/) the pin is honoured automatically and the interpreter is
downloaded if missing:

```sh
uv venv                                      # creates .venv on Python 3.12 (git-ignored)
                                             # without uv: python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt           # litellm, duckdb, pandas
uv pip install -r requirements-embed-st.txt  # embedding server (MiniLM for text + CLIP for images)
uv pip install -r requirements-lotus.txt     # only to run the LOTUS baselines (step 6)
```

(`pip install -r ...` works the same without uv.) Activate the environment in every shell that runs
`serve/start_stack.sh` or a harness — they use whatever `python3` / `litellm` is on PATH.

The embedding server downloads its two models from Hugging Face on first start (about 1.2 GB into
`~/.cache/huggingface`): `sentence-transformers/all-MiniLM-L6-v2` for text predicates and
`clip-ViT-B-32` for image predicates. They feed the selectivity model that orders AI predicates; the server
runs them on the CPU. Without an embedding server every predicate is treated as equally selective: results
are unchanged, call counts can be higher. `requirements-embed-mlx.txt` is an optional Apple-Silicon
alternative through `mlx-embeddings` (`AI_EMBED_BACKEND=mlx`), untested against the recorded runs.

## 4. Start the serving stack

The engine speaks plain http to an OpenAI-compatible endpoint. The default setup chains a cache proxy
(records every answer with its latency and cost and replays it on repeat) in front of litellm (talks to
OpenAI), plus the embedding server:

```sh
serve/start_stack.sh                         # litellm :4000, cache proxy :4001, embedding server :4002
```

It prints one line per service and the proxy's cache statistics when everything is up. Stop with
`pkill -f ai_cache_server.py; pkill -f 'litellm --config'; pkill -f ai_embed_server.py`.

## 5. Run a query

Open the SemBench MOVIE database that ships in the repo (2,000 Rotten Tomatoes reviews of 105 movies;
`reviews(id, criticName, reviewText, scoreSentiment, …)`, `movies(id, title, director, genre, …)`):

```sh
build/reldebug/duckdb -readonly sembench/MOVIE/movie.db
```

```sql
SET ai_endpoint = 'http://localhost:4001';   -- the cache proxy; the built-in default is litellm at :4000
SET ai_model = 'gpt-5.6-luna';                -- the default

-- a semantic filter: which reviews of Booksmart are clearly positive?  (5 reviews -> 5 calls, 3 pass)
SELECT criticName, left(reviewText, 60) || '...' AS review
FROM reviews
WHERE id = 'booksmart'
  AND ai_filter('Determine if the following movie review is clearly positive. Review: "' || reviewText || '".');

-- a semantic classification, checked against the dataset's own sentiment label
SELECT scoreSentiment,
       ai_classify('Classify the sentiment of this review as either ''POSITIVE'' or ''NEGATIVE''. Review: '
                   || reviewText, ['POSITIVE', 'NEGATIVE']) AS predicted,
       count(*) AS n
FROM reviews WHERE id = 'baby_driver'
GROUP BY ALL ORDER BY ALL;                    -- POSITIVE / POSITIVE / 5

SELECT llm_calls, cache_hits, failed_calls, cost_usd FROM ai_usage();   -- one row per query above
```

Drop the `WHERE id = …` to run over all 2,000 reviews: the region operator folds duplicate prompts,
orders predicates by learned selectivity and keeps 20 requests in flight; `ai_usage()` shows the calls.

`ai_filter(prompt) → BOOLEAN`, `ai_classify(input, [labels]) → VARCHAR`, `ai_score(input, rubric[, lo, hi])`,
`ai_complete(prompt) → VARCHAR`, `ai_agg(list, instruction)`, `ai_embed(text)`, and `ai_image(path)` to put an
image into a prompt. Prompts are ordinary SQL string expressions, so any column can be concatenated in.

**If a function returns `NULL` and `ai_usage()` shows `llm_calls = 0`**, the request never got an answer:
`failed_calls` counts such requests and the CLI prints one `[aisql] LLM request to <endpoint> failed: …`
line with the reason (typically nothing listening on the endpoint — start `serve/start_stack.sh`, or
`SET ai_endpoint` to where litellm or the proxy runs). Failed answers are never cached, so the next query
retries. `SET ai_max_retries` (default 6, exponential backoff) bounds the wait.

## 6. SemBench MOVIE: SWAN vs LOTUS

The MOVIE suite (10 queries over the reviews above) is fully self-contained in the repo: data, queries,
gold answers and the recorded results of both systems. Every call goes through the cache proxy, so a run
that matches the recorded prompts replays answers, latency and cost for free; a first run with a different
model or prompt pays the provider.

**SWAN** (about 18 minutes replayed; $1.53 recorded):

```sh
cd sembench/MOVIE
python3 swan_movie.py --serial              # all 10 queries, one at a time
python3 swan_movie.py --serial q1 q7        # a subset
```

Each query prints latency, LLM calls (`fresh` = not served from the cache), tokens, cost and its quality
metric; the last line is the suite's macro quality. Everything is written to `swan_movie_results.json`
(`SWAN_TAG=_x` suffixes the file names so an experiment never overwrites the baseline). The runner finds
`../../build/reldebug/duckdb` on its own; set `DUCKDB_BIN` for another build.

**LOTUS** (the official SemBench LOTUS queries, same model, same proxy; about 3 hours replayed because the
proxy replays LOTUS's recorded latency too; $14.45 recorded):

```sh
python3 lotus_movie.py                      # all 10 queries
python3 lotus_movie.py q1 q7                # a subset
```

It needs `requirements-lotus.txt` from step 3 and writes `lotus_movie_results.json`. To replay without the
recorded latency, start the proxy with `CACHE_SIMULATE_LATENCY=0 serve/start_stack.sh` (calls and cost stay
exact; latency then measures only the local machine).

**Compare** (per query and in total; works on the committed results before you run anything):

```sh
cd ..                                       # sembench/
python3 compare.py MOVIE
```

| q | SWAN quality | LOTUS quality | SWAN calls | LOTUS calls | SWAN lat (s) | LOTUS lat (s) | SWAN $ | LOTUS $ |
|---|---|---|---|---|---|---|---|---|
| … | | | | | | | | |
| **macro / Σ** | **0.818** | **0.780** | 18,806 | 201,344 | 1139 | 10654 | 1.53 | 14.45 |

Quality is SemBench's own metric per query (F1, count/ratio accuracy, Spearman for the ranking queries).
ECOMM and MMQA run the same way (`ECOMM/swan_ecomm.py --serial`, `MMQA/swan_mmqa.py`, `compare.py ECOMM`),
but most of their queries need the image assets, which are not in git — see `sembench/README.md`.

## Settings

All settings are `SET`-able per connection; the `AI_*` environment variable in the last column seeds the
default when set (the bench harnesses use them). `SELECT name, value FROM duckdb_settings() WHERE name LIKE 'ai_%'`
shows the live values.

**Connection and model**

| setting | values | default | what it does |
|---|---|---|---|
| `ai_endpoint` | URL (http) | `http://localhost:4000` | OpenAI-compatible chat endpoint: litellm directly, or `http://localhost:4001` for the cache proxy (`AI_PROXY_URL`) |
| `ai_model` | model id | `gpt-5.6-luna` | chat model for every AI function (`AI_MODEL`) |
| `ai_api_key` | string | *(empty)* | bearer token sent to `ai_endpoint`; empty for the local stack (`AI_API_KEY`) |
| `ai_reasoning_effort` | `low` / `medium` / `high` / *(empty)* | *(empty = omit)* | forwarded per request when set (`AI_REASONING_EFFORT`) |
| `ai_concurrency` | integer ≥ 1 | `20` | in-flight LLM requests, process-wide (`AI_MAX_CONCURRENCY`) |
| `ai_max_retries` | integer ≥ 0 | `6` | retries per request on 429/503/529 or a transient failure, exponential backoff (`AI_MAX_RETRIES`) |
| `ai_hedge` | `true` / `false` | `true` | duplicate a call still unanswered past the observed p99 latency; first answer wins (`AI_HEDGE`) |
| `ai_http_keepalive` | `true` / `false` | `true` | reuse HTTP connections across requests (`AI_HTTP_KEEPALIVE`) |
| `ai_prefix_cache` | `true` / `false` | `true` | explicit provider prompt caching for factor-graph pair prompts (`AI_PREFIX_CACHE`) |
| `ai_local_cache` | `true` / `false` | `true` | in-process response cache for chat + embeddings |
| `ai_local_cache_scope` | `query` / `cross_query` | `query` | `query`: one query never serves another (the benchmark rule); `cross_query`: process lifetime |
| `ai_embed_endpoint` | URL (http) | `http://localhost:4002` | embeddings server for the selectivity model (`AI_EMBED_URL`) |
| `ai_embed_model` | model id | `sentence-transformers/all-MiniLM-L6-v2` | model name sent with each embeddings request; the server embeds with whatever it loaded (`AI_EMBED_MODEL`; the server's image model is `AI_EMBED_IMAGE_MODEL`) |
| `ai_embed_images` | `true` / `false` | `true` | embed image predicates (predicate text × image through CLIP) for the selectivity model; `false`, or a text-only server, keeps a neutral prior for them (`AI_EMBED_IMAGES`) |
| `ai_embed_concurrency` | integer ≥ 1 | `4` | embedding requests in flight at once (`AI_EMBED_CONCURRENCY`) |
| `ai_embed_batch_images` | integer ≥ 1 | `8` | image items per embedding request (`AI_EMBED_BATCH_IMAGES`) |

**TypeSafe System One (Jev) backend** — optional, per function

| setting | values | default | what it does |
|---|---|---|---|
| `ai_typesafe` | csv of `filter`, `classify` | *(empty = off)* | route `ai_filter` → Noul and/or `ai_classify` → Choice to TypeSafe; image-bearing prompts stay on the chat model (`AI_TYPESAFE`) |
| `ai_typesafe_endpoint` | URL (http) | `http://localhost:4001` | the cache proxy, which terminates TLS towards `https://api.typesafe.ai` (`AI_TYPESAFE_URL`) |
| `ai_typesafe_model` | model id | `jev-latest` | (`AI_TYPESAFE_MODEL`) |
| `ai_typesafe_api_key` | string | *(empty)* | (`TYPESAFE_API_KEY`) |
| `ai_typesafe_threshold` | 0.0 – 1.0 | `0.5` | Noul probability at or above which `ai_filter` is true (`AI_TYPESAFE_THRESHOLD`) |

**Optimizer** — every stage is result-preserving; each can be switched off independently

| setting | values | default | what it does |
|---|---|---|---|
| `ai_inline_ai_ctes` | `true` / `false` | `true` | inline CTEs that contain AI functions so pruning and pull-up can reach the predicate |
| `ai_semi_reduce` | `true` / `false` | `true` | Yannakakis semi-join reduction before any AI evaluation |
| `ai_pullup` | `true` / `false` | `true` | lift semantic filters above the joins (`DUCKDB_SEMANTIC_PULLUP`) |
| `ai_reorder` | `true` / `false` | `true` | ordering of AI predicates with learned selectivity + speculative evaluation (`DUCKDB_AI_REORDER`) |
| `ai_factorize` | `off` / `filters` / `all` | `all` | AI region placement: none / above AI filters only / every AI call (`DUCKDB_AI_DEDUP`, `DUCKDB_AI_SCAN_REGION`) |
| `ai_join_factorize` | `off` / `pushdown` / `factor` | `factor` | AI-condition joins: expand / push the region below the join / factor graph over the pair domain (`DUCKDB_AI_GROUP_JOIN`) |
| `ai_limit` | `true` / `false` | `true` | LIMIT push-down into AI evaluation (early stop) (`DUCKDB_AI_LIMIT`) |

**Debug / experiment knobs** (`ai_debug_*`) — stable but not part of the user contract

| setting | values | default | what it does |
|---|---|---|---|
| `ai_debug_log` | csv of `region`, `spec`, `yann`, `leaftexts`, `mock` | *(empty)* | stderr diagnostics per subsystem (`region` prints the leaf-region timers) |
| `ai_debug_wave_overlap` | integer ≥ 1 | `8` | region waves kept in flight at once (1 = synchronous) |
| `ai_debug_wave_size` | integer | `0` | override the region wave floor (0 = 5 × `ai_concurrency`) |
| `ai_debug_embed_slice` | integer | `100` | rows the region ingests per slice before it embeds, decides and fires (`AI_EMBED_SLICE`; 0 = the whole chunk) |
| `ai_debug_embed_slice_text` | `true` / `false` | `true` | slice text-only regions too (`AI_EMBED_SLICE_TEXT`); `false` slices only regions with an image predicate |
| `ai_debug_warm_gate` | `true` / `false` | `true` | after a region's first slice is decided cold, wait until every predicate has a batch of verdicts and the model has trained before deciding more rows (`AI_WARM_GATE`) |
| `ai_debug_region_blocking` | `true` / `false` | `false` | disable the region's streaming sink (materialize, then evaluate) |
| `ai_debug_graph_eval` | `staged` / `lazy` / `lazy-adaptive` | `lazy-adaptive` | factor-graph scheduler |
| `ai_debug_no_train` | `true` / `false` | `false` | freeze the selectivity model (no training from verdicts) |
| `ai_debug_mlp_seed` | integer | `0` | seed for the selectivity model's init and warm-up picks |
| (env) `AI_MLP_FIFO` / `AI_MLP_MIN_TRAIN` | integer | `256` / `16` | the selectivity model's training window (most recent labelled examples) and the examples a step needs; read once at load |
| `ai_debug_speculative_always` | `true` / `false` | `true` | drop the speculative fan-out gate |
| `ai_debug_speculative_min_fanout` | 0.0 – 1.0 | `0.3` | minimum fan-out to add a speculative node |
| `ai_debug_speculative_threshold` | 0.0 – 1.0 | `0.5` | speculative pass-through threshold |
| `ai_debug_trust_image_estimate` | `true` / `false` | `false` | let speculative all-image rows trust the estimate instead of always evaluating |
| `ai_debug_prompt_variant` | `strict` / `soft` / `plain` | `strict` | `ai_filter` system prompt; `plain` sends the bare prompt (cross-engine, prompt-identical comparisons) |
| `ai_debug_semi_reduce_force` | `true` / `false` | `false` | semi-reduce every join cluster, AI or not (A/B benchmarking) |
| `ai_debug_embed_filter` | `true` / `false` | `false` | embedding pre-filter pass |
| `ai_debug_agg_distinct` | `true` / `false` | `false` | allow `ai_agg` over DISTINCT inputs (changes the multiset) |

Embedding-server environment (`serve/ai_embed_server.py`): `AI_EMBED_MODEL` (text model), `AI_EMBED_IMAGE_MODEL`
(CLIP model; empty = text only), `AI_EMBED_DEVICE` (default `cpu`), `AI_EMBED_PORT`, `AI_EMBED_BACKEND` (`st` / `mlx`).
`GET http://localhost:4002/` reports what it loaded.

## Hybrid bench (agent_bench Q1–Q30, vs PLOP)

```sh
CACHE_ALIAS_CHAT=1 serve/start_stack.sh        # agent_bench replays need the proxy's shared-verdict alias (see sembench/README.md)
cd sembench/AGENTBENCH
python3 swan_agentbench.py                     # Q1..Q30, ~14 min replayed (Q19 is 10 of them)
python3 swan_agentbench.py Q16 Q17             # a subset
python3 eval_agentbench.py swan plop           # quality vs the LIMIT-free PLOP ground truth (results/plop_gt)
SWAN_TAG=swan python3 report_three_way.py      # per-query SWAN / PLOP / LOTUS table -> results/agentbench_comparison_three_way.md
```

The SWAN translations are `swan_queries/Q*.sql`, PLOP's originals `plop_queries/`, and the PLOP-side runs
(`results/plop`, `results/plop_gt`, `results/plop_none`) are committed so scoring needs no PLOP install.
Runs are tagged (`SWAN_TAG=my_run` → `results/my_run/` + `results/my_run_agentbench_results.json`) and
scored by tag: `python3 eval_agentbench.py my_run plop`.

## Layout

```
src/                 engine (C++): client/ model/ functions/ rep/ exec/ plan/ optimizer/ mock/
src/include/         headers
test/sql/            sqllogictests (registered with the unittest binary via LOAD_TESTS)
bench/               call-count / latency correctness benches (python, mock or proxy)
serve/               start_stack.sh, ai_cache_server.py, ai_embed_server.py, litellm.config.yaml
sembench/            SemBench MOVIE / ECOMM / MMQA and agent_bench (SWAN, PLOP, LOTUS queries), harnesses,
                     ground truth, recorded results, compare.py
docs/DESIGN.md       the design: two currencies, pipeline, boundary rule, duplication safety, caching, measured state
```
