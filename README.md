# SWAN AI-SQL — semantic SQL for DuckDB

`ai_filter / ai_classify / ai_score / ai_complete / ai_agg / ai_embed / ai_image` for DuckDB, plus an
optimizer and executor built around **factorized ("two-currency") execution**: every LLM call runs once per
*distinct* input, joins over AI predicates evaluate the distinct *pair domain* without materializing the
cross product, and learned predicate ordering, Yannakakis semi-join reduction and LIMIT early-stop minimize
calls. Benchmarked against LOTUS on SemBench (MOVIE / ECOMM / MMQA) and, on the 30-query hybrid bench the
PLOP authors shared with us, against PLOP's recorded runs and LOTUS. The design is in
[docs/DESIGN.md](docs/DESIGN.md).

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

The `duckdb/` submodule is pinned to the DuckDB commit the engine is written against. No vcpkg dependencies
are needed: the LLM client uses DuckDB's bundled http-only `httplib`, and the local proxies in `serve/`
terminate TLS towards the providers.

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
TYPESAFE_API_KEY=...        # only for SET ai_typesafe='filter,classify,score' (ai_filter/ai_classify/ai_score on TypeSafe Jev)
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
Every answer is structured output (a JSON schema per function); `ai_filter` has the model state its reason
in one sentence before the verdict, which measurably sharpens borderline factual claims (SWAN benchmark
0.75 → 0.78 mean quality, about 35 more output tokens per call).

**Writing a filter prompt.** Keep the question apart from the evidence. When the context is a short fact
about a named entity (a name, an address), this layout measured best on the SWAN benchmark at the same cost;
on long free-text contexts (SemBench's movie reviews) the one-line form above does as well or better:

```sql
ai_filter('Context:' || chr(10) || '[school_address]: «' || school_address || '»' || chr(10) || chr(10) || chr(10)
          || 'Claim: Is the school in the city of Fresno? school_address')
```

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
| **macro / Σ** | **0.832** | **0.780** | 18,804 | 201,344 | 1686 | 10654 | 2.28 | 14.45 |

Quality is SemBench's own metric per query (F1, count/ratio accuracy, Spearman for the ranking queries).
ECOMM and MMQA run the same way (`ECOMM/swan_ecomm.py --serial`, `MMQA/swan_mmqa.py`, `compare.py ECOMM`),
but most of their queries need the image assets, which are not in git — see `sembench/README.md`.

## Settings

Everything is a DuckDB setting (`SET ai_... = ...`, per connection). The ones most users touch:

| setting | default | what it does |
|---|---|---|
| `ai_endpoint` | `http://localhost:4000` | the OpenAI-compatible chat endpoint (`http://localhost:4001` for the cache proxy) |
| `ai_model` | `gpt-5.6-luna` | the chat model for every AI function |
| `ai_api_key` | *(empty)* | bearer token for the endpoint; empty for the local stack |
| `ai_concurrency` | `20` | LLM requests in flight |
| `ai_embed_endpoint` | `http://localhost:4002` | the embedding server behind the selectivity model |

Every optimizer stage has an on/off setting (`ai_semi_reduce`, `ai_pullup`, `ai_reorder`, `ai_factorize`,
`ai_join_factorize`, `ai_limit`, ...) for parity tests and A/B runs; the defaults are the measured
composition. The full list, with every default and the environment variable that seeds it, is in
[docs/SETTINGS.md](docs/SETTINGS.md).

## Hybrid bench (agent_bench Q1–Q30): SWAN vs LOTUS, with PLOP as the reference

The 30 hybrid queries (relational plans with semantic operators over DataAgentBench and TPC-H data) are
the benchmark introduced by the PLOP paper. PLOP itself is not publicly released, so it cannot be rerun
here; its authors shared the queries and their execution results with us, and both are committed under
`sembench/AGENTBENCH/` (`sembench/README.md` describes the ground truth). What you can run is SWAN and
LOTUS on the same queries and compare all three:

```sh
cd sembench/AGENTBENCH
python3 swan_agentbench.py                     # SWAN Q1..Q30 (~14 min; a first run pays ~$0.30 to the provider)
python3 swan_agentbench.py Q16 Q17             # a subset
python3 lotus_agentbench.py                    # LOTUS Q1..Q30 (our translations, lotus_queries/; ~1 h, ~$3)
python3 eval_agentbench.py swan lotus plop     # quality of each vs the LIMIT-free ground truth
SWAN_TAG=swan python3 report_three_way.py      # per-query SWAN / PLOP / LOTUS table -> results/agentbench_comparison_three_way.md
```

The SWAN translations are `swan_queries/Q*.sql` (`translate_to_swan.py` documents the mapping). Runs are
tagged (`SWAN_TAG=my_run` → `results/my_run/` + `results/my_run_agentbench_results.json`) and scored by
tag: `python3 eval_agentbench.py my_run plop`.

Read the quality column with the sampling in mind: the published SWAN row reused PLOP's recorded model
verdicts for every shared prompt, while a fresh run samples the model anew, so its agreement with the
ground truth also measures verdict variance (`sembench/README.md` rule 7). Calls, cost and latency are
comparable either way.

## Acknowledgements

- The hybrid benchmark and PLOP's execution results were shared by the PLOP authors — Qiuyang Mang,
  Yufan Xiang, Hangrui Zhou, Runyuan He, Jiaxiang Yu, Hanchen Li, Aditya Parameswaran and Alvin Cheung,
  *PLOP: Cost-Based Placement of Semantic Operators in Hybrid Query Plans* (arXiv:2604.09944, 2026).
  Thank you for making the comparison possible.
- SemBench's MOVIE / ECOMM / MMQA suites and its official LOTUS runners are used as published; the
  LOTUS baselines run on [LOTUS](https://github.com/lotus-data/lotus) unmodified.

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
docs/SETTINGS.md     every setting, its default and the environment variable that seeds it
```
