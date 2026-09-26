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

The serving stack and the benchmark harnesses are Python (≥ 3.11). Use a virtual environment; with
[uv](https://docs.astral.sh/uv/):

```sh
uv venv                                      # creates .venv (git-ignored); or: python3 -m venv .venv
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
SET ai_endpoint = 'http://localhost:4001';           -- the cache proxy (default)
SET ai_model = 'gpt-5.6-luna';                        -- the default
SELECT title FROM movies WHERE ai_filter('This review is clearly positive: ' || review);
SELECT * FROM ai_usage();                             -- calls, cache hits, tokens, cost per query
```

Settings: `ai_endpoint / ai_model / ai_api_key / ai_concurrency / ai_embed_endpoint / ai_embed_model`;
optimizer toggles `ai_factorize (off/filters/all)`, `ai_join_factorize (off/pushdown/factor)`, `ai_reorder`,
`ai_pullup`, `ai_limit`, `ai_semi_reduce`, `ai_local_cache`; `ai_typesafe` routes `ai_filter` / `ai_classify`
to TypeSafe System One (Jev) with `ai_typesafe_endpoint / ai_typesafe_model / ai_typesafe_api_key /
ai_typesafe_threshold`; and an `ai_debug_*` expert namespace. `AI_MODEL`, `AI_PROXY_URL`,
`AI_MAX_CONCURRENCY` and the other `AI_*` environment variables seed the defaults.

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
