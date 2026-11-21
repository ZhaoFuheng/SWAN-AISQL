# SWAN AISQL

SWAN AISQL introduces the `ai_filter / ai_classify / ai_score / ai_complete / ai_agg / ai_embed / ai_image`
semantic operators to DuckDB, with an optimizer and executor built around **factorized ("two-currency")
execution**: every LLM call runs once per *distinct* input, joins over AI predicates evaluate the distinct
*pair domain* without materializing the cross product, and learned predicate ordering, Yannakakis semi-join
reduction and LIMIT early-stop minimize calls. It is benchmarked against LOTUS, BlendSQL, Palimpzest
and PLOP ([Morrila](https://github.com/DMoRanSky/Morrila), the authors' DuckDB fork) on SemBench
(MOVIE / ECOMM / MMQA), on the 30-query hybrid bench of the PLOP paper, and on
[SWAN 2.0](https://github.com/ZhaoFuheng/SWANBench). The design is in [docs/DESIGN.md](docs/DESIGN.md).

## Quick start

```sh
pip install "swan-aisql[embed]"        # the extension, the DuckDB it runs in, and the local embeddings server
export OPENAI_API_KEY=sk-...
```

```python
import swan_aisql

con = swan_aisql.connect()             # OpenAI directly, model gpt-5.6-luna; any DuckDB connection otherwise
con.sql("""
    CREATE TABLE reviews AS SELECT * FROM (VALUES
        ('A. Lane',  'A tender, funny film that earns every one of its tears.'),
        ('B. Ruiz',  'Two hours I will not get back; the plot collapses by the second act.'),
        ('C. Okafor', 'Gorgeous to look at, and the lead performance is the best of the year.'),
        ('D. Meyer', 'Competent but forgettable, the kind of sequel nobody asked for.')
    ) v(critic, review)
""")
con.sql("""
    SELECT critic, review
    FROM reviews
    WHERE ai_filter('Is this movie review clearly positive? Review: ' || review)
""").show()                            # the two positive reviews, one model call per row
con.sql("SELECT llm_calls, cache_hits FROM ai_usage()").show()
```

`connect()` returns an ordinary DuckDB connection with the AI functions loaded. The notebook
[python/examples/swan2_formula_1.ipynb](python/examples/swan2_formula_1.ipynb) does this on a SWAN 2.0 benchmark
database: five of its questions, each with the model calls it cost, and a LIMIT push-down comparison.
[python/README.md](python/README.md) has the package details (endpoints, settings, the embeddings server).

## Results

Every system runs with `gpt-5.6-luna`, 20 requests in flight, every call through a cache proxy that records
the answer, its latency and its cost. The *SWAN-AISQL (Jev)* rows are the one exception: the same engine under
[`SET ai_typesafe`](docs/SETTINGS.md), where TypeSafe's Jev answers the text filters and classifications (on MOVIE
also the bounded scores) and `gpt-5.6-luna` the rest ([per query](aisql-bench/typesafe_comparison_20261120.md)).
Quality is each benchmark's own metric; calls are model requests; cost is
the provider's price for them; latency is wall-clock over the suite's queries run one after the other. Every
row is a replay of the recorded calls on one machine (a MacBook Air M5, November 2026): one system at a time, the
proxy returning each recorded answer after its recorded response time, with no provider reachable. The
protocol behind each table, and how to rerun every system, is in [aisql-bench/README.md](aisql-bench/README.md).

**SemBench MOVIE** (text sentiment over reviews, 10 queries)

| | quality | calls | cost | latency |
|---|---|---|---|---|
| LOTUS 1.2.4 | 0.780 | 201k | $14.45 | 10,579 s |
| BlendSQL 0.1.27 | 0.718† | 18.7k | $1.44 | 1,163 s |
| Palimpzest 1.5.3 (Abacus optimizer) | 0.761† | 68.5k | $17.88 | 4,778 s |
| PLOP-DP | 0.729† | 19.2k | $1.70 | 1,649 s |
| **SWAN-AISQL** | **0.818** | 18.8k | $2.25 | 1,771 s |
| SWAN-AISQL (Jev) | 0.795 | 18.8k | $0.39 | 458 s |

†Mean of eleven replays: on MOVIE's unordered LIMIT queries these systems return whichever rows arrive first,
so a single replay varies.

**SemBench ECOMM** (14 queries over fashion product listings with photos) and **MMQA** (11 queries over
tables that mix text and images). BlendSQL and PLOP take no images.

| ECOMM | quality | calls | cost | latency |
|---|---|---|---|---|
| LOTUS 1.2.4 | 0.639 | 17.8k | $6.41 | 1,546 s |
| Palimpzest 1.5.3 (Abacus optimizer) | 0.672 | 18.2k | $16.76 | 1,397 s |
| **SWAN-AISQL** | **0.832** | 15.8k | $6.74 | 1,448 s |
| SWAN-AISQL (Jev) | 0.814 | 15.7k | $5.77 | 907 s |

| MMQA | quality | calls | cost | latency |
|---|---|---|---|---|
| LOTUS 1.2.4 | 0.450 | 19.0k | $2.43 | 1,391 s |
| Palimpzest 1.5.3 (Abacus optimizer) | 0.687 | 46.3k | $13.79 | 3,244 s |
| **SWAN-AISQL** | 0.648 | 16.5k | $2.41 | 1,346 s |
| SWAN-AISQL (Jev) | 0.599 | 14.1k | $1.93 | 1,060 s |

**Hybrid bench** (the 30 agent_bench queries of the PLOP paper, from the authors' public fork
[Morrila](https://github.com/DMoRanSky/Morrila): relational plans with semantic operators)

| | macro-F1 | calls | cost | latency |
|---|---|---|---|---|
| LOTUS 1.2.4 | 0.632* | 25,741 | $2.90 | 1,789 s |
| BlendSQL 0.1.27 | 0.714* | 25,441 | $2.91 | 1,954 s |
| Palimpzest 1.5.3 (Abacus optimizer) | 0.735* | 25,777 | $6.82 | 2,040 s |
| PLOP-DP | 1.000 | 13,328 | $0.50 | 905 s |
| **SWAN-AISQL** | **1.000** | **10,997** | **$0.28** | **731 s** |

SWAN and PLOP consume the same recorded verdict per shared prompt, so both match the ground truth (PLOP's
un-rewritten execution) by construction; *the other systems' verdicts are independent samples, so their
quality also measures verdict variance, while their calls, cost and latency are comparable.

**SWAN 2.0** (120 questions over four BIRD databases scaled to ~10k rows; one AISQL query per question that
every system plans itself)

| | quality | calls | cost | latency |
|---|---|---|---|---|
| LOTUS 1.2.4 | 0.752 | 69,204 | $4.04 | 4,026 s |
| BlendSQL 0.1.27 | 0.761 | 59,620 | $4.95 | 4,961 s |
| Palimpzest 1.5.3 (Abacus optimizer) | 0.778 | 70,925 | $9.81 | 5,202 s |
| PLOP-DP | 0.698 | 25,639 | $1.11 | 10,208 s |
| **SWAN-AISQL** | 0.777 | **18,751** | **$2.01** | **1,927 s** |

Quality is at parity among LOTUS, BlendSQL, Palimpzest and SWAN (the model's verdicts bound it); calls, cost
and latency are the separation.

**Reproducing the tables.** The recorded answers of every system on every benchmark live in one store,
`serve/.llm_cache.duckdb` (2.5 M recorded calls, 4.7 GB), which the cache proxy serves to any of them, so a
replay reproduces a table at zero cost and needs no key. The store is published on Zenodo
([10.5281/zenodo.23112764](https://doi.org/10.5281/zenodo.23112764); the DOI resolves to the latest version, which
is the one behind the tables above); `serve/fetch_cache.sh` downloads, checks and unpacks it (1.0 GB download). The
other systems' residual run-to-run differences (random comparison order in LOTUS's ranking operators, result assembly
in BlendSQL and Palimpzest, which rows an unordered LIMIT keeps in BlendSQL, Palimpzest and PLOP) are theirs,
not the cache's.

## Build

```sh
git clone --recurse-submodules https://github.com/ZhaoFuheng/SWAN-AISQL
cd SWAN-AISQL
GEN=ninja make reldebug        # or: make release / make debug   (~25 min, builds DuckDB too)
make test_reldebug             # the sqllogictests, on an in-process mock LLM: no network, no key
```

This builds `build/reldebug/duckdb`, a CLI with the extension linked in, the loadable
`aisql.duckdb_extension`, and the test runner, against the DuckDB v2.0.0 LTS the `duckdb/` submodule is pinned
to. TLS, the version string the PyPI wheel requires, packaging and the code-quality checks are in
[docs/GUIDE.md](docs/GUIDE.md).

## Using it

`ai_filter(prompt) → BOOLEAN`, `ai_classify(input, [labels]) → VARCHAR`, `ai_score(input, rubric[, lo, hi])`,
`ai_complete(prompt[, json_schema]) → VARCHAR`, `ai_agg(list, instruction)`, `ai_embed(text)`, and `ai_image(path)`
to put an image into a prompt. Prompts are ordinary SQL string expressions, so any column can be concatenated
in; every answer but `ai_complete`'s is structured output. Keys, the local serving stack (cache proxy, litellm,
embedding server), a first query from the CLI, how to write a filter prompt and what a `NULL` answer means are in
[docs/GUIDE.md](docs/GUIDE.md). Everything is a DuckDB setting, per connection, with every optimizer stage
switchable for parity runs: [docs/SETTINGS.md](docs/SETTINGS.md) lists them all.

## Acknowledgements

- The hybrid benchmark and the Morrila fork are the work of the PLOP authors — Qiuyang Mang,
  Yufan Xiang, Hangrui Zhou, Runyuan He, Jiaxiang Yu, Hanchen Li, Aditya Parameswaran and Alvin Cheung,
  *PLOP: Cost-Based Placement of Semantic Operators in Hybrid Query Plans* (arXiv:2604.09944, 2026),
  who shared the queries with us before releasing them. Thank you for making the comparison possible.
- SemBench's MOVIE / ECOMM / MMQA suites and its official LOTUS runners are used as published; the
  LOTUS baselines run on [LOTUS](https://github.com/lotus-data/lotus) unmodified.
