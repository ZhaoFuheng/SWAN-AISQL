# SWAN AISQL

SWAN AISQL introduce `ai_filter / ai_classify / ai_score / ai_complete / ai_agg / ai_embed / ai_image` semantic operators, plus an
optimizer and executor built around **factorized ("two-currency") execution**: every LLM call runs once per
*distinct* input, joins over AI predicates evaluate the distinct *pair domain* without materializing the
cross product, and learned predicate ordering, Yannakakis semi-join reduction and LIMIT early-stop minimize
calls. Benchmarked against LOTUS on SemBench (MOVIE / ECOMM / MMQA), against PLOP's recorded runs and LOTUS
on the 30-query hybrid bench the PLOP authors shared with us, and against BlendSQL and LOTUS on
[SWAN 2.0](https://github.com/ZhaoFuheng/SWANBench). The design is in [docs/DESIGN.md](docs/DESIGN.md).

This repository is a DuckDB **extension** built on the
[extension template](https://github.com/duckdb/extension-template): the engine is the `aisql` extension,
DuckDB is the `duckdb/` submodule, CI comes from `extension-ci-tools/`, and dependencies are declared in
`vcpkg.json` (OpenSSL, so the LLM client can speak https; optional for a local build).

The results come first; then the numbered sections are a path: build, keys, Python environment, start the
serving stack, run a query, then run SemBench MOVIE for SWAN and LOTUS and compare them. Reference material
(settings, the hybrid bench, SWAN 2.0, layout) follows.

## Results

All three comparisons use `gpt-5.6-luna` for every system, 20 requests in flight, every call through the
cache proxy (the recorded answers are published, see Keys). Quality is each benchmark's own metric; calls are
chat requests; cost is the provider's price for them.

**SemBench MOVIE** (sf2000: text sentiment over reviews, 10 queries; quality = the suite's macro F1 / count
accuracy / Spearman)

| | quality | calls | cost | latency (10 q) |
|---|---|---|---|---|
| LOTUS 1.2.4 | 0.780 | 201k | $14.45 | 10,654 s |
| BlendSQL 0.1.27 | 0.763 | 18.7k | $1.44 | 1,188 s |
| Palimpzest 1.5.3 (Abacus optimizer) | 0.776 | 68.5k | $17.89 | 4,865 s |
| PLOP-DP | 0.726 | 19.1k | $1.69 | 1,665 s |
| **SWAN-AISQL** | **0.832** | 18.8k | $2.28 | 1,686 s |

Latency is wall-clock over the suite's queries run one after the other. Palimpzest runs SemBench's own
Palimpzest programs for the suite under its Abacus optimizer; its latency is the execution time it reports
per query, without the start-up of its own interpreter.
SemBench's two image suites, ECOMM and MMQA, run SWAN against LOTUS (PLOP and BlendSQL do not support
images); their tables are in section 6.

**Hybrid bench** (the 30 agent_bench queries of the PLOP paper: relational plans with semantic operators;
quality = row-multiset F1 against the LIMIT-free PLOP ground truth; SWAN and PLOP consume the same prompt template)

| | macro-F1 | calls | cost | latency |
|---|---|---|---|---|
| LOTUS 1.2.4 | 0.607* | 25,749 | $2.90 | 1,802 s |
| BlendSQL 0.1.27 | 0.758* | 24,631 | $2.76 | 1,960 s |
| Palimpzest 1.5.3 (Abacus optimizer) | 0.710* | 25,785 | $6.82 | 2,088 s |
| ThalamusDB 0.1.15 | 0.661* | 19,512 | $2.09 | 8,542 s |
| PLOP-DP | 1.000 | 13,602 | $0.53 | 1,080 s |
| **SWAN-AISQL** | **1.000** | **11,172** | **$0.29** | **799 s** |

*LOTUS's, BlendSQL's, ThalamusDB's and Palimpzest's verdicts are independent samples, so where an answer
hinges on a few judgments any disagreement scores 0; their calls, cost and latency are comparable. Latency is
wall-clock over the 30 queries run one after the other, 20 requests in flight. BlendSQL runs our mechanical
translation of the SWAN queries (`aisql-bench/AGENTBENCH/translate_to_blendsql.py`). ThalamusDB runs our
formulation of the queries
(`aisql-bench/AGENTBENCH/thalamusdb_queries/`, rule in `make_thalamusdb_queries.py`): each semantic predicate
becomes an `NLfilter` over a materialised column carrying the prompt's fields, as SemBench did for its own
ThalamusDB queries, and the classification of Q1–Q3 becomes a CASE cascade of such filters, one per
category (specific classes first, the broadest as the fallback). Palimpzest runs the LOTUS programs
unchanged with their semantic operators served by Palimpzest (`aisql-bench/palimpzest_hybrid_exec.py`), its
Abacus optimizer choosing each operator's plan.

**SWAN 2.0** (120 questions over four BIRD databases scaled to ~10k rows with duplicate entities; one AISQL
query per question that every system plans itself; quality = relative error / F1, as in SemBench)

| | quality | exact | calls | cost | latency (120 q) |
|---|---|---|---|---|---|
| LOTUS 1.2.4 | 0.760 | 59/120 | 69,204 | $4.04 | 4,101 s |
| BlendSQL 0.1.27 | 0.761 | 60/120 | 59,620 | $4.95 | 4,608 s |
| Palimpzest 1.5.3 (Abacus optimizer) | 0.766 | 60/120 | 70,933 | $9.85 | 2,991 s |
| PLOP-DP | 0.690 | 51/120 | 25,604 | $1.72 | 10,431 s |
| **SWAN-AISQL** | 0.763 | 60/120 | **22,327** | **$2.42** | **2,331 s** |

The quality column is parity among LOTUS, BlendSQL, Palimpzest and SWAN (the model's verdicts bound it);
the calls, cost and latency columns are the separation. Section 6 and the two benchmark sections after the settings
say how to reproduce each table.

## 1. Build

```sh
git clone --recurse-submodules https://github.com/ZhaoFuheng/SWAN-AISQL
cd SWAN-AISQL
GEN=ninja make reldebug        # or: make release / make debug   (~25 min, builds DuckDB too)
```

- `build/reldebug/duckdb` — the CLI with `aisql` (plus `parquet`, `json`, `tpch`) linked in
- `build/reldebug/extension/aisql/aisql.duckdb_extension` — the loadable extension
- `build/reldebug/test/unittest` — the test runner with the extension's sqllogictests registered

The `duckdb/` submodule is pinned to the DuckDB commit the engine is written against: a DuckDB 2.0 pre-release,
the commit the `duckdb` 2.0 nightly wheel on PyPI is built from, so the Python package below loads into that
wheel. If CMake finds OpenSSL (Homebrew's `openssl@3`, or `libssl-dev` / the vcpkg port in CI) the LLM client
speaks https directly and `SET ai_endpoint = 'https://api.openai.com'` works; without it the build is http-only
and the local proxies in `serve/` terminate TLS towards the providers, which is how every benchmark here runs.

To load the loadable extension you build into the `duckdb` wheel from PyPI (rather than into the CLI and test
runner above, which do not care), the build has to carry that wheel's version string: `SELECT version()` in it
(`v2.0.0-alpha43763` for the pinned pre-release) goes into the configure step as
`DUCKDB_VERSION=v2.0.0-alpha43763 GEN=ninja make reldebug`. `python/build_wheel.sh` also accepts a plain build
and re-stamps it.

**Python instead of a build.** `pip install "swan-aisql[embed]"` installs a wheel with the extension for your
platform, the pinned `duckdb` and the embeddings server; `swan_aisql.connect()` returns a DuckDB connection with
the functions loaded, talking to OpenAI with `OPENAI_API_KEY` from the environment and starting the embeddings
server on demand ([python/README.md](python/README.md)). The wheels come out of the `Python Wheels` workflow; from a checkout,
`python/build_wheel.sh` packs the binary you just built.

To check the build, run the test suite. It uses an in-process deterministic mock LLM, so it needs no
network or keys:

```sh
make test_reldebug                                      # every sqllogictest under test/sql/ (python3 >= 3.10 on PATH)
build/reldebug/test/unittest "*test/sql/region/*"       # one group
```

CI also runs DuckDB's code-quality checks on every push. Locally: `make format-check` (or `format-fix`) needs
clang-format 11.0.1 on PATH (`pip install clang-format==11.0.1`), and `make tidy-check` needs clang-tidy
(Homebrew's `llvm` on macOS, where the Makefile also passes the SDK to the tidy build).

## 2. Keys

Copy `.env.example` to `.env` (git-ignored) and fill in what you use:

```
OPENAI_API_KEY=sk-...       # required: the chat model (default gpt-5.6-luna) via litellm
TYPESAFE_API_KEY=...        # only for SET ai_typesafe='filter,classify,score' (ai_filter/ai_classify/ai_score on TypeSafe Jev)
```

`serve/start_stack.sh` and the benchmark harnesses read `.env`; nothing else in the repo carries a key.

**No key? Replay the published cache.** Every number in this README was produced through the cache proxy,
and the recorded answers are published on Zenodo ([10.5281/zenodo.23112764](https://doi.org/10.5281/zenodo.23112764); a 225 MB download that unpacks to about 1.6 GB; the DOI resolves to the latest version). `serve/fetch_cache.sh` downloads
them to `serve/.llm_cache.duckdb`; the stack then starts without litellm, and SemBench, the hybrid bench and
SWAN 2.0 replay their answers, latency and cost at $0 — for SWAN and for the baselines (LOTUS on every
benchmark, PLOP and BlendSQL where they run), so every comparison table reproduces, the SWAN 2.0 latency
session included. Model inference at temperature 0 is treated as deterministic: the published cache is the
single source of answers and latencies for every system, each table is a replay of it, and asking the
provider afresh is a different experiment rather than a rerun. Replay matches requests byte for byte, so it covers
the repository's queries with the default prompts and model; an edited query is a fresh request and needs
a key. One caveat: which prompts the engine sends depends on run-time decisions — under a `LIMIT` it stops
as soon as it has enough rows, and the learned ordering of the predicates follows the answers as they arrive —
so a replay can touch a few prompts the recording never did; without a key those calls fail and the query
answers from the rows it could decide (`ai_usage()` shows them as `failed_calls`). The published cache was
therefore converged: SWAN-AISQL's SWAN 2.0 run was replayed nine times with the provider reachable, each
pass recording the prompts it newly touched (573 the first time, then 159 down to 40 per pass out of 22.4k,
with the answer to every question unchanged throughout). A replay without a key can still meet a few
unrecorded prompts: the published SWAN 2.0 table is such a replay, which met 70 (0.3%, on 6 questions),
returned the recorded answer to every question, and took 2,331 s against the live run's 2,045 s (the
retries of those prompts); the SemBench and hybrid-bench replays met none. ThalamusDB samples which rows
to judge next and Palimpzest does not issue the same prompts from run to run, so their replays can meet
more: a Palimpzest replay of SWAN 2.0 met 2,148 unrecorded prompts and a ThalamusDB replay lost five
questions to them. Their rows are therefore their live runs, served from the cache where the prompts
repeat, and an exact rerun of either needs a key.

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
alternative through `mlx-embeddings` (`AI_EMBED_BACKEND=mlx`), untested against the recorded runs. The server's
code is the pip package's `swan_aisql.embed_server`; `serve/ai_embed_server.py` runs it from the checkout.

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

## 6. SemBench: MOVIE for all systems, ECOMM and MMQA for SWAN vs LOTUS

The MOVIE suite (10 queries over the reviews above) is fully self-contained in the repo: data, queries,
gold answers and the recorded results of all four systems. Every call goes through the cache proxy, so a run
that matches the recorded prompts replays answers, latency and cost for free; a first run with a different
model or prompt pays the provider.

**SWAN** (about 28 minutes replayed; $2.28 recorded):

```sh
cd aisql-bench/MOVIE
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

**PLOP and BlendSQL** on the same ten queries (`plop_queries/`, `blendsql_queries/`), scored the same way:

```sh
python3 plop_movie.py                       # PLOP_BIN = the Morrila fork's shell (aisql-bench/PLOP_FORK.md: the three edits it needs)
<python with blendsql> blendsql_movie.py    # e.g. ../../../SWAN_bench/.venv/bin/python
```

Neither supports images, so ECOMM and MMQA are not run for them.

**ThalamusDB** (PyPI `thalamusdb`, the version SemBench evaluated) on SemBench's own ThalamusDB formulations of
the queries (`thalamusdb_queries/`, copied from the SemBench repository), run as SemBench's runner runs them:
exact mode (error bound 0, call and token caps lifted, 6,000 s per query), 20 requests in flight. Its dialect
has boolean `NLfilter` / `NLjoin` predicates only, so SemBench wrote eight of the ten MOVIE queries for it
(q9 and q10 rank). It runs in its own Python environment:

```sh
../setup_thalamusdb.sh                      # once: aisql-bench/.venv-thalamusdb with thalamusdb 0.1.15 (or set THALAMUSDB_PYTHON)
python3 thalamusdb_movie.py                 # eight queries; ../ECOMM/thalamusdb_ecomm.py and ../MMQA/thalamusdb_mmqa.py likewise
```

**Palimpzest** (PyPI `palimpzest` 1.5.3, MIT DSG's semantic-operator system) on SemBench's own Palimpzest
programs (`palimpzest_queries.py`, copied from the SemBench repository; all ten MOVIE queries), under
SemBench's Palimpzest settings (MaxQuality, parallel execution, 20 workers) with its Abacus optimizer's
pareto plan search, which SemBench had off; Palimpzest's sample-based cost estimation runs only with a
validator or training set, which the suites do not supply, so the default cost model decides. Its calls
and tokens are the proxy's counter deltas while a query runs, so run it with nothing else using the proxy;
cost is computed from those tokens at list price. It runs in its own
Python environment:

```sh
../setup_palimpzest.sh                      # once: aisql-bench/.venv-palimpzest with palimpzest 1.5.3 (or set PALIMPZEST_PYTHON)
python3 palimpzest_movie.py                 # ten queries; ../ECOMM/palimpzest_ecomm.py (q1-q13) and ../MMQA/palimpzest_mmqa.py (all eleven) likewise
```

**Compare** (per query and in total; works on the committed results before you run anything):

```sh
cd ..                                       # aisql-bench/
python3 compare.py MOVIE
```

It prints one row per query with each recorded system's quality (SemBench's own metric per query: F1,
count/ratio accuracy, Spearman for the ranking queries), calls, latency and cost, and a macro row that is
the MOVIE table above.

**ECOMM and MMQA** (the image suites: ECOMM sf500, 14 queries over fashion product listings with photos; MMQA sf200,
11 queries over tables that mix text and images; quality = macro F1 / count accuracy / ARI) run the same way
(`ECOMM/swan_ecomm.py --serial`, `MMQA/swan_mmqa.py`, `compare.py ECOMM`), but most of their queries need
the image assets, which are not in git — see `aisql-bench/README.md`. PLOP and BlendSQL do not support images,
so these suites compare SWAN with LOTUS on the full suite and with Palimpzest on SemBench's programs for it
(ThalamusDB ran the few queries SemBench wrote for it; see below):

**ECOMM** (14 q)

| | quality | calls | cost | latency (14 q) |
|---|---|---|---|---|
| LOTUS 1.2.4 | 0.637 | 17.8k | $6.40 | 1,990 s |
| Palimpzest 1.5.3 (Abacus optimizer) | 0.680 on q1–q13 | 17.7k | $16.33 | 1,494 s |
| **SWAN-AISQL** | **0.724** | 16.4k | $8.32 | 2,882 s |

**MMQA** (11 q)

| | quality | calls | cost | latency (11 q) |
|---|---|---|---|---|
| LOTUS 1.2.4 | 0.449 | 19.0k | $2.43 | 1,720 s |
| Palimpzest 1.5.3 (Abacus optimizer) | 0.687 | 46.3k | $13.79 | 3,284 s |
| **SWAN-AISQL** | **0.692** | 15.9k | $2.31 | 1,314 s |

Quality is each suite's own metric and latency is wall-clock over its queries. Palimpzest runs SemBench's
own Palimpzest programs, which exist for ECOMM q1–q13 (on those 13, SWAN scores 0.703 and LOTUS 0.648) and
all of MMQA; its cost is computed from its tokens and its latency is the execution time it reports, without
its interpreter's start-up. ThalamusDB covers only the queries SemBench wrote for it (five of ECOMM's
fourteen, seven of MMQA's eleven; the rest need map, classify or ranking operators it lacks), too few for a
comparable macro, so its numbers stay in the per-query tables: `compare.py ECOMM`, `compare.py MMQA`.

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

## Hybrid bench (agent_bench Q1–Q30): SWAN vs PLOP, LOTUS, BlendSQL, ThalamusDB and Palimpzest

The 30 hybrid queries (relational plans with semantic operators over DataAgentBench and TPC-H data) are
the benchmark introduced by the PLOP paper. PLOP itself is not publicly released, so it cannot be rerun
here; its authors shared the queries and their execution results with us, and both are committed under
`aisql-bench/AGENTBENCH/` (`aisql-bench/README.md` describes the ground truth). What you can run is SWAN and
LOTUS and BlendSQL on the same queries and compare all four:

```sh
CACHE_ALIAS_CHAT=1 serve/start_stack.sh        # shared-verdict mode: SWAN's bare prompts key on the text PLOP sent, so
                                               # SWAN replays PLOP's recorded verdicts (aisql-bench/README.md, rule 7)
cd aisql-bench/AGENTBENCH
python3 swan_agentbench.py                     # SWAN Q1..Q30 (~14 min; a first run pays $0.30 to the provider)
python3 swan_agentbench.py Q16 Q17             # a subset
python3 lotus_agentbench.py                    # LOTUS Q1..Q30 (our translations, lotus_queries/; ~1 h, $3)
<python with blendsql> blendsql_agentbench.py  # BlendSQL Q1..Q30 (translate_to_blendsql.py; ~35 min, $3)
python3 thalamusdb_agentbench.py               # ThalamusDB Q1..Q30 (thalamusdb_queries/, our formulation; ../setup_thalamusdb.sh first)
python3 palimpzest_agentbench.py               # Palimpzest Q1..Q30 (the LOTUS programs, operators served by Palimpzest; ../setup_palimpzest.sh first)
python3 eval_agentbench.py swan lotus plop blendsql thalamusdb palimpzest   # quality of each vs the LIMIT-free ground truth
SWAN_TAG=swan python3 report_three_way.py      # per-query table of every system -> results/agentbench_comparison_three_way.md
```

The SWAN translations are `swan_queries/Q*.sql` (`translate_to_swan.py` documents the mapping). Runs are
tagged (`SWAN_TAG=my_run` → `results/my_run/` + `results/my_run_agentbench_results.json`) and scored by
tag: `python3 eval_agentbench.py my_run plop`.

The runner sends `ai_filter`'s prompt bare (`AI_PROMPT_VARIANT=plain`: no system message, PLOP's exact
text), which is what lets the proxy's alias serve PLOP's recorded verdict to SWAN; without the alias, or
with the default prompt variant, a run samples the model anew and its agreement with the ground truth also
measures verdict variance (`aisql-bench/README.md` rule 7). Calls, cost and latency are comparable either way.

## SWAN 2.0: SWAN vs BlendSQL, LOTUS, PLOP and Palimpzest

[SWAN 2.0](https://github.com/ZhaoFuheng/SWANBench) is a 120-question benchmark over four BIRD databases (scaled to
~10k rows and duplicated): one AISQL query per question, which every system runs as written — SWAN-AISQL
directly, BlendSQL and LOTUS through mechanical translations — so each system's planner decides the LLM
calls. Its `scripts/run_swan_aisql.sh` clones and builds this repository, starts the serving stack and runs
all 120 questions; the quality score follows SemBench (relative error for numbers, F1 for row sets).

| system | quality (mean) | exact | LLM calls | cost | latency (120 q) |
|---|---|---|---|---|---|
| LOTUS 1.2.4 | 0.760 | 59/120 | 69,204 | $4.04 | 4,101 s |
| BlendSQL 0.1.27 | 0.761 | 60/120 | 59,620 | $4.95 | 4,608 s |
| Palimpzest 1.5.3 (Abacus optimizer) | 0.766 | 60/120 | 70,933 | $9.85 | 2,991 s |
| PLOP-DP (Morrila fork, our translation) | 0.690 | 51/120 | 25,604 | $1.72 | 10,431 s |
| **SWAN-AISQL** | 0.763 | 60/120 | **22,327** | **$2.42** | **2,331 s** |

gpt-5.6-luna for all five, zero-shot, 20 requests in flight, each system recorded fresh through an empty
recording cache and replayed from the published cache with the recorded latencies, so the numbers
reproduce. The benchmark also runs
ThalamusDB, whose dialect expresses 69 of the 120 questions (its README has that row). The model's
verdicts bound the quality column (the benchmark's results README gives the earlier recordings), so read
the calls, cost and latency columns as the clear separation. Per-question answers, scores and seconds:
`results/gpt-5.6-luna/` in that repository. The 120 queries, their oracle forms and the question list are
also kept here under `aisql-bench/SWAN2/` (a copy of the benchmark's, refreshed by its `sync.sh`), so this
repository holds every query behind its tables; the databases and the harness stay in the benchmark.

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
aisql-bench/            SemBench MOVIE / ECOMM / MMQA and agent_bench: queries, harnesses and recorded results for
                     SWAN, LOTUS, PLOP, BlendSQL, ThalamusDB and Palimpzest, ground truth, compare.py, PLOP_FORK.md
                     (running the Morrila fork), setup_thalamusdb.sh / setup_palimpzest.sh (their own environments);
                     SWAN2/ = a copy of the SWAN 2.0 benchmark's 120 queries, oracle forms and question list
python/              the swan-aisql pip package: swan_aisql.connect(), build_wheel.sh, the wheel smoke test
docs/DESIGN.md       the design: two currencies, pipeline, boundary rule, duplication safety, caching, measured state
docs/SETTINGS.md     every setting, its default and the environment variable that seeds it
```
