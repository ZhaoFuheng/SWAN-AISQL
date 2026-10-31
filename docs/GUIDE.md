# Building and using SWAN AISQL

How to build the extension from source and check it, then how to use the AI functions: keys, the local serving
stack, a first query, prompts, `NULL` answers and the settings people touch most.

## Building, testing and packaging

```sh
git clone --recurse-submodules https://github.com/ZhaoFuheng/SWAN-AISQL
cd SWAN-AISQL
GEN=ninja make reldebug        # or: make release / make debug   (~25 min, builds DuckDB too)
```

- `build/reldebug/duckdb` — the CLI with `aisql` (plus `parquet`, `json`, `tpch`) linked in
- `build/reldebug/extension/aisql/aisql.duckdb_extension` — the loadable extension
- `build/reldebug/test/unittest` — the test runner with the extension's sqllogictests registered

The repository is a DuckDB extension built on the [extension template](https://github.com/duckdb/extension-template):
the engine is the `aisql` extension, DuckDB is the `duckdb/` submodule, CI comes from `extension-ci-tools/`, and
dependencies are declared in `vcpkg.json`. The submodule is pinned to the DuckDB release the engine is written
against, the v2.0.0 LTS tag, the same build as the `duckdb` 2.0.0 wheel on PyPI. [UPDATING.md](UPDATING.md) is the
template's recipe for moving the pin.

**TLS.** If CMake finds OpenSSL (Homebrew's `openssl@3`, or `libssl-dev` / the vcpkg port in CI) the LLM client
speaks https directly and `SET ai_endpoint = 'https://api.openai.com'` works; without it the build is http-only and
the local proxies in `serve/` terminate TLS towards the providers, which is how every benchmark runs.

**Version string.** To load the extension you build into the `duckdb` wheel from PyPI (rather than into the CLI
and test runner above, which do not care), the build has to carry that wheel's version string, `v2.0.0`. A plain
build of the tag reports a development version, so configure with
`OVERRIDE_GIT_DESCRIBE=v2.0.0 GEN=ninja make reldebug`. `python/build_wheel.sh` also accepts a plain build and
re-stamps it.

**The pip package.** `swan-aisql` on PyPI is this extension prebuilt for each platform, pinned to the `duckdb`
wheel it loads into. The wheels come out of the `Python Wheels` workflow; from a checkout, `python/build_wheel.sh`
packs the binary you just built. [../python/README.md](../python/README.md) has the package details.

**Tests.** The suite uses an in-process deterministic mock LLM, so it needs no network or keys:

```sh
make test_reldebug                                      # every sqllogictest under test/sql/ (python3 >= 3.10 on PATH)
build/reldebug/test/unittest "*test/sql/region/*"       # one group
```

One case pins a DuckDB optimizer off: the DISTINCT ON barrier in `test/sql/pullup/semi_reduce_barriers.test` runs
with `SET disabled_optimizers = 'join_filter_pushdown'`. DuckDB v2.0.0's join filter pushdown routes a hash join's
runtime filter through a `DISTINCT ON` onto the probe side's scan, and the scan applies it only when the join's
build finishes before the scan starts, so a `DISTINCT ON` subquery joined on its result can return a different
representative from run to run (0.5-6% of runs at 8 threads for the 10-row table in that test on a from-source
build, load dependent; no AI function is involved, the test's comment has the plain-SQL repro). Upstream main fixed it on 2026-10-29 (commit 782a53071c66, the
`LOGICAL_DISTINCT` case now recurses only for a plain `DISTINCT`); the v2.0 LTS branch does not carry it yet.

**Code quality.** CI runs DuckDB's checks on every push. Locally, `make format-check` (or `format-fix`) needs
clang-format 11.0.1 (`pip install clang-format==11.0.1`) and a Python new enough for its tooling
(`PYTHON=/path/to/python3.11+ make format-check` when the default `python3` is older), and `make tidy-check`
needs clang-tidy (Homebrew's `llvm` on macOS, where the Makefile also passes the SDK to the tidy build).

## The AI functions

`ai_filter(prompt) → BOOLEAN`, `ai_classify(input, [labels]) → VARCHAR`, `ai_score(input, rubric[, lo, hi])`,
`ai_complete(prompt[, json_schema]) → VARCHAR`, `ai_agg(list, instruction)`, `ai_embed(text)`, and `ai_image(path)`
to put an image into a prompt. Prompts are ordinary SQL string expressions, so any column can be concatenated in.
Every answer but `ai_complete`'s is structured output (a JSON schema per function; `ai_complete(prompt)` returns
the raw completion, and `ai_complete(prompt, json_schema)` enforces the schema you pass); `ai_filter` has the
model state its reason in one sentence before the verdict, which measurably sharpens borderline factual claims
(SWAN benchmark 0.75 → 0.78 mean quality, about 35 more output tokens per call).

### Keys

Copy `.env.example` to `.env` (git-ignored) and fill in what you use:

```
OPENAI_API_KEY=sk-...       # required: the chat model (default gpt-5.6-luna) via litellm
TYPESAFE_API_KEY=...        # only for SET ai_typesafe='filter,classify,score' (ai_filter/ai_classify/ai_score on TypeSafe Jev)
```

`serve/start_stack.sh` and the benchmark harnesses read `.env`; nothing else in the repo carries a key. Without a
key the recorded cache replays every benchmark; see [the benchmark README](../aisql-bench/README.md).

### The serving stack

The engine speaks plain http to an OpenAI-compatible endpoint. The default setup chains a cache proxy (records
every answer with its latency and cost and replays it on repeat) in front of litellm (talks to OpenAI), plus the
embedding server that feeds the selectivity model. It needs the Python environment described in the benchmark
README (`uv venv`, `requirements.txt`, `requirements-embed-st.txt`):

```sh
serve/start_stack.sh                         # litellm :4000, cache proxy :4001, embedding server :4002
```

It prints one line per service and the proxy's cache statistics when everything is up, and reuses a service it
finds running, so stop one first to change its mode: `pkill -f ai_cache_server.py; pkill -f 'litellm --config';
pkill -f ai_embed_server.py`. The embedding server downloads its two models from Hugging Face on first start
(about 1.2 GB into `~/.cache/huggingface`): `sentence-transformers/all-MiniLM-L6-v2` for text predicates and
`clip-ViT-B-32` for image predicates, run on the CPU. Without an embedding server every predicate is treated as
equally selective: results are unchanged, call counts can be higher. `requirements-embed-mlx.txt` is an optional
Apple-Silicon alternative through `mlx-embeddings` (`AI_EMBED_BACKEND=mlx`), untested against the recorded runs.
The server's code is the pip package's `swan_aisql.embed_server`; `serve/ai_embed_server.py` runs it from the
checkout.

### A first query from the CLI

Open the SemBench MOVIE database that ships in the repo (2,000 Rotten Tomatoes reviews of 105 movies;
`reviews(id, criticName, reviewText, scoreSentiment, …)`, `movies(id, title, director, genre, …)`):

```sh
build/reldebug/duckdb -readonly aisql-bench/MOVIE/movie.db
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

Drop the `WHERE id = …` to run over all 2,000 reviews: the region operator folds duplicate prompts, orders
predicates by learned selectivity and keeps 20 requests in flight; `ai_usage()` shows the calls.

### Writing a filter prompt

Keep the question apart from the evidence. When the context is a short fact about a named entity (a name, an
address), this layout measured best on the SWAN benchmark at the same cost; on long free-text contexts
(SemBench's movie reviews) the one-line form above does as well or better:

```sql
ai_filter('Context:' || chr(10) || '[school_address]: «' || school_address || '»' || chr(10) || chr(10) || chr(10)
          || 'Claim: Is the school in the city of Fresno? school_address')
```

### If a function returns NULL

If `ai_usage()` shows `llm_calls = 0` alongside, the request never got an answer: `failed_calls` counts such
requests and the CLI prints one `[aisql] LLM request to <endpoint> failed: …` line with the reason (typically
nothing listening on the endpoint — start `serve/start_stack.sh`, or `SET ai_endpoint` to where litellm or the
proxy runs). Failed answers are never cached, so the next query retries. `SET ai_max_retries` (default 6,
exponential backoff) bounds the wait. A request that gets no answer is a `NULL` result on every path alike (a
scalar call, the region's dispatcher, a factor graph), so a filter drops the row, `NOT ai_filter(...)` never
passes it, and `ai_filter(...) IS NULL` is true for it. A `NULL` prompt is a `NULL` answer too, without a call.

### Settings

Everything is a DuckDB setting (`SET ai_... = ...`). The endpoint, model, key, retries and prices, and every
optimizer flag, are per connection (`SET GLOBAL` changes the process default); the request pool, the embedding
client and the local cache are process-wide. The ones most users touch:

| setting | default | what it does |
|---|---|---|
| `ai_endpoint` | `http://localhost:4000` | the OpenAI-compatible chat endpoint (`http://localhost:4001` for the cache proxy) |
| `ai_model` | `gpt-5.6-luna` | the chat model for every AI function |
| `ai_api_key` | *(empty)* | bearer token for the endpoint; empty for the local stack |
| `ai_concurrency` | `20` | LLM requests in flight |
| `ai_embed_endpoint` | `http://localhost:4002` | the embedding server behind the selectivity model |

Every optimizer stage has an on/off setting (`ai_semi_reduce`, `ai_pullup`, `ai_reorder`, `ai_factorize`,
`ai_join_factorize`, `ai_limit`, ...) for parity tests and A/B runs; the defaults are the measured composition.
The full list, with every default and the environment variable that seeds it, is in [SETTINGS.md](SETTINGS.md).
