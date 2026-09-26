# SWAN AI-SQL — semantic SQL for DuckDB

`ai_filter / ai_classify / ai_score / ai_complete / ai_agg / ai_embed / ai_image` for DuckDB, plus an
optimizer and executor built around **factorized ("two-currency") execution**: every LLM call runs once per
*distinct* input, joins over AI predicates evaluate the distinct *pair domain* without materializing the
cross product, and per-row DP ordering, Yannakakis semi-join reduction, speculative pruning and LIMIT
early-stop minimize calls. Benchmarked against LOTUS on SemBench (MOVIE / ECOMM / MMQA) and against PLOP on
the hybrid bench (agent_bench Q1–Q30); the design and the recorded numbers are in [docs/DESIGN.md](docs/DESIGN.md).

This repository is a DuckDB **extension** built on the
[extension template](https://github.com/duckdb/extension-template): the engine is the `aisql` extension,
DuckDB is the `duckdb/` submodule, CI comes from `extension-ci-tools/`, and dependencies are declared in
`vcpkg.json` (none required today).

## Build

```sh
git clone --recurse-submodules https://github.com/ZhaoFuheng/SWAN-AISQL
cd SWAN-AISQL
GEN=ninja make reldebug        # or: make release / make debug   (~25 min, builds DuckDB too)
```

- `build/reldebug/duckdb` — the CLI with `aisql` (plus `parquet`, `json`, `tpch`) linked in
- `build/reldebug/extension/aisql/aisql.duckdb_extension` — the loadable extension
- `build/reldebug/test/unittest` — the test runner with the extension's sqllogictests registered

The submodules track DuckDB `main` (pinned to the commit the code is written against; the engine uses
current planner/vector APIs that are not in the v1.5 release line). `vcpkg` is optional: the Makefile only
uses it when `VCPKG_TOOLCHAIN_PATH` is set, and `vcpkg.json` lists no dependencies (the LLM client uses
DuckDB's bundled http-only `httplib` and `yyjson`; TLS towards providers is terminated by the local proxies
in `serve/`).

## Test

```sh
make test_reldebug                                      # every sqllogictest under test/sql/ (python3 >= 3.10 on PATH)
build/reldebug/test/unittest "*test/sql/region/*"       # one group
```

Tests run against the in-process deterministic mock LLM (`CALL ai_mock_start()`), so they need no network
or keys.

## Keys

Copy `.env.example` to `.env` (git-ignored) and fill in what you use:

```
OPENAI_API_KEY=sk-...       # required: the chat model (default gpt-5.6-luna) and embeddings via litellm
TYPESAFE_API_KEY=...        # only for SET ai_typesafe='filter,classify' (ai_filter/ai_classify on TypeSafe Jev)
```

`serve/start_stack.sh` and the bench harnesses read `.env`; nothing else in the repo carries a key.

## Python environment

The serving stack and the benchmark harnesses are Python; `.python-version` pins **3.12** (any ≥ 3.11
works). Use a virtual environment; with [uv](https://docs.astral.sh/uv/) the pin is honoured
automatically and the interpreter is downloaded if missing:

```sh
uv venv                                      # creates .venv on Python 3.12 (git-ignored)
                                             # without uv: python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt           # litellm, duckdb, pandas
uv pip install -r requirements-embed-mlx.txt # embedding server, Apple Silicon (text + image)  -- or:
uv pip install -r requirements-embed-st.txt  # embedding server, any host (text only)
uv pip install -r requirements-lotus.txt     # only to run the LOTUS baselines
```

(`pip install -r ...` works the same without uv.) Activate the environment in every shell that runs
`serve/start_stack.sh` or a harness — they use whatever `python3` / `litellm` is on PATH.

**Embedding model.** `serve/ai_embed_server.py` serves one dual-encoder model for text *and* images.
On Apple Silicon it loads `mlx-community/clip-vit-base-patch32` through `mlx-embeddings`; elsewhere it
falls back to `sentence-transformers/all-MiniLM-L6-v2` (text only — image leaves then get a neutral
selectivity prior). Either model is downloaded from Hugging Face on the server's first start
(~600 MB / ~90 MB, cached under `~/.cache/huggingface`); no manual install step. Override with
`AI_EMBED_MODEL=<hf id>` (mlx) or `AI_EMBED_TEXT_FALLBACK=<hf id>` (sentence-transformers), and point
the engine elsewhere with `SET ai_embed_endpoint`. `serve/embed.log` shows which backend came up.

## Serving stack

The engine speaks plain http to an OpenAI-compatible endpoint. The default setup chains a cache proxy
(records every answer with its latency and cost, replays it on repeat, forwards TypeSafe calls) in front of
litellm (talks to OpenAI):

```sh
serve/start_stack.sh                         # litellm :4000, cache proxy :4001, embedding server :4002
```

The embedding server feeds the selectivity model that orders AI predicates; without it every predicate is
treated as equally selective (results are unchanged, call counts can be higher). `--no-embed` skips it.

## Use

```sql
SET ai_endpoint = 'http://localhost:4001';           -- the cache proxy; the built-in default is litellm at :4000
SET ai_model = 'gpt-5.6-luna';                        -- the default
SELECT title FROM movies WHERE ai_filter('This review is clearly positive: ' || review);
SELECT * FROM ai_usage();                             -- calls, cache hits, failed calls, tokens, cost per query
```

**If a function returns `NULL` and `ai_usage()` shows `llm_calls = 0`**, the request never got an answer:
`failed_calls` counts such requests and the CLI prints one `[aisql] LLM request to <endpoint> failed: …`
line with the reason (typically nothing listening on the endpoint — start `serve/start_stack.sh`, or
`SET ai_endpoint` / `AI_PROXY_URL` to where litellm or the proxy runs). Failed answers are never cached,
so the next query retries. `SET ai_max_retries` (default 6, exponential backoff) bounds the wait.

### Settings

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
| `ai_embed_model` | model id | `sentence-transformers/all-MiniLM-L6-v2` | model name sent to the embeddings server (`AI_EMBED_MODEL`; `serve/start_stack.sh` serves the CLIP dual encoder) |

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
| `ai_reorder` | `true` / `false` | `true` | DP ordering of AI predicates with learned selectivity + speculative evaluation (`DUCKDB_AI_REORDER`) |
| `ai_factorize` | `off` / `filters` / `all` | `all` | AI region placement: none / above AI filters only / every AI call (`DUCKDB_AI_DEDUP`, `DUCKDB_AI_SCAN_REGION`) |
| `ai_join_factorize` | `off` / `pushdown` / `factor` | `factor` | AI-condition joins: expand / push the region below the join / factor graph over the pair domain (`DUCKDB_AI_GROUP_JOIN`) |
| `ai_limit` | `true` / `false` | `true` | LIMIT push-down into AI evaluation (early stop) (`DUCKDB_AI_LIMIT`) |

**Debug / experiment knobs** (`ai_debug_*`) — stable but not part of the user contract

| setting | values | default | what it does |
|---|---|---|---|
| `ai_debug_log` | csv of `region`, `spec`, `yann`, `leaftexts`, `mock` | *(empty)* | stderr diagnostics per subsystem (`region` prints the leaf-region timers) |
| `ai_debug_wave_overlap` | integer ≥ 1 | `8` | region waves kept in flight at once (1 = synchronous) |
| `ai_debug_wave_size` | integer | `0` | override the region wave floor (0 = 5 × `ai_concurrency`) |
| `ai_debug_region_blocking` | `true` / `false` | `false` | disable the region's streaming sink (materialize, then evaluate) |
| `ai_debug_graph_eval` | `staged` / `lazy` / `lazy-adaptive` | `lazy-adaptive` | factor-graph scheduler |
| `ai_debug_no_train` | `true` / `false` | `false` | freeze the selectivity model (no training from verdicts) |
| `ai_debug_mlp_seed` | integer | `0` | seed for the selectivity model's init and warm-up picks |
| `ai_debug_speculative_always` | `true` / `false` | `true` | drop the speculative fan-out gate |
| `ai_debug_speculative_min_fanout` | 0.0 – 1.0 | `0.3` | minimum fan-out to add a speculative node |
| `ai_debug_speculative_threshold` | 0.0 – 1.0 | `0.5` | speculative pass-through threshold |
| `ai_debug_trust_image_estimate` | `true` / `false` | `false` | let speculative all-image rows trust the estimate instead of always evaluating |
| `ai_debug_prompt_variant` | `strict` / `soft` / `plain` | `strict` | `ai_filter` system prompt; `plain` sends the bare prompt (cross-engine, prompt-identical comparisons) |
| `ai_debug_semi_reduce_force` | `true` / `false` | `false` | semi-reduce every join cluster, AI or not (A/B benchmarking) |
| `ai_debug_embed_filter` | `true` / `false` | `false` | embedding pre-filter pass |
| `ai_debug_agg_distinct` | `true` / `false` | `false` | allow `ai_agg` over DISTINCT inputs (changes the multiset) |

## Benchmarks

All harnesses need the Python environment and the serving stack above. Every call goes through the cache proxy, so a
rerun replays recorded answers, latency and cost for free; a first run pays the provider.

### SemBench MOVIE

```sh
cd sembench/MOVIE
DUCKDB_BIN=../../build/reldebug/duckdb python3 swan_movie.py --serial        # all 10 queries, ~18 min replayed
DUCKDB_BIN=../../build/reldebug/duckdb python3 swan_movie.py --serial q7     # one query
```

Each query prints latency, LLM calls (`fresh` = not served from the cache), tokens, cost and its quality
metric; the suite's macro-quality is the last line, and everything is written to
`swan_movie_results.json` (`SWAN_TAG=_x` suffixes the file names so an experiment never overwrites the
baseline). `python3 lotus_movie.py` is the LOTUS side of the comparison. ECOMM and MMQA run the same way
(`ECOMM/swan_ecomm.py --serial`, `MMQA/swan_mmqa.py`) but most of their queries need the image assets,
which are not in git — see `sembench/README.md`.

### Hybrid bench (agent_bench Q1–Q30, vs PLOP)

```sh
cd sembench/AGENTBENCH
DUCKDB_BIN=../../build/reldebug/duckdb python3 swan_agentbench.py            # Q1..Q30, ~14 min replayed (Q19 is 10 of them)
DUCKDB_BIN=../../build/reldebug/duckdb python3 swan_agentbench.py Q16 Q17    # a subset
python3 eval_agentbench.py swan plop            # quality vs the LIMIT-free PLOP ground truth (results/plop_gt)
SWAN_TAG=swan python3 report_agentbench.py      # per-query table: quality, calls, latency, cost -> results/agentbench_comparison.md
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
                     ground truth, recorded results and comparison tables
docs/DESIGN.md       the design: two currencies, pipeline, duplication safety, caching, measured state
```
