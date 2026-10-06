# swan-aisql

SWAN AI-SQL as a pip package: DuckDB with semantic SQL functions (`ai_filter`, `ai_classify`, `ai_score`,
`ai_complete`, `ai_agg`, `ai_embed`, `ai_image`) and the SWAN optimizer, which folds duplicate prompts,
orders predicates by learned selectivity, pushes `LIMIT` into the model calls and keeps a fixed number of
requests in flight. Engine, design and benchmarks: <https://github.com/ZhaoFuheng/SWAN-AISQL>.

```sh
pip install swan-aisql
```

```python
import os, swan_aisql

con = swan_aisql.connect(api_key=os.environ["OPENAI_API_KEY"])      # OpenAI directly, model gpt-5.6-luna
con.sql("CREATE TABLE reviews AS SELECT * FROM 'reviews.parquet'")
con.sql("""
    SELECT critic, review
    FROM reviews
    WHERE ai_filter('Is this movie review clearly positive? Review: ' || review)
    LIMIT 10
""").show()
con.sql("SELECT llm_calls, cache_hits, cost_usd FROM ai_usage()").show()
```

`connect()` returns a plain `duckdb.DuckDBPyConnection`; `swan_aisql.load(con)` adds the extension to a
connection you opened yourself (with `allow_unsigned_extensions`).

**Endpoint.** Any OpenAI-compatible chat endpoint: `connect(endpoint="https://api.openai.com")`, a litellm
proxy, or the repository's caching proxy (`endpoint="http://localhost:4001"`). With no `endpoint`, the
extension uses `AI_PROXY_URL` from the environment and, failing that, OpenAI when `OPENAI_API_KEY` is set.
`https://` endpoints need the TLS build of the extension, which the published wheels are. Requests carry
`temperature: 0`; a model that rejects the parameter (the gpt-5 reasoning models answer HTTP 400 to it)
gets the request again without it, and the process leaves it out from then on.

**Settings.** `model`, `api_key`, `reasoning_effort` and `price_input` / `price_output` / `price_cached` (USD per 1M
tokens, so `ai_usage().cost_usd` prices a direct endpoint's calls) are keywords; every other `SET ai_*` option passes
through, e.g. `connect(ai_concurrency=50, ai_limit=False)`. The full list is `SELECT * FROM
duckdb_settings() WHERE name LIKE 'ai_%'` and the repository README.

**Embeddings.** The learned predicate ordering embeds prompts and inputs. Any OpenAI-compatible embeddings
endpoint serves: `connect(ai_embed_endpoint="https://api.openai.com", ai_embed_model="text-embedding-3-small")`
uses OpenAI's (text only), and the repository's `serve/ai_embed_server.py` (`pip install sentence-transformers`)
is the local server every published number used, which also embeds images for `ai_image` predicates. Without
one the engine notices on the first request, orders predicates by prompt cost alone from then on, and every
query still runs.

**Version coupling.** A DuckDB extension loads only into the exact DuckDB build it was compiled for, so each
swan-aisql release pins one `duckdb` version (this release: the version in the package metadata, a 2.0
pre-release). `connect()` reports a mismatch with the `pip install 'duckdb==...'` line that fixes it.

**Building a wheel yourself** (from the repository checkout, after `make release`):

```sh
pip install build wheel --pre "duckdb==$(sed -n 's/^dependencies = \["duckdb==\([^"]*\)"\]/\1/p' python/pyproject.toml)"
python/build_wheel.sh                        # packs build/release/extension/aisql/aisql.duckdb_extension
pip install python/dist/swan_aisql-*.whl
```

The binary must come from the DuckDB commit the pinned `duckdb` wheel was built from (the `duckdb/` submodule
pin); the script checks the binary's footer against the installed engine and refuses any other build.

**Releasing** (maintainers). `.github/workflows/PythonWheels.yml` builds the extension for every platform
through the DuckDB distribution pipeline, packs one wheel per platform, smoke-tests each against the pinned
`duckdb` wheel on its own runner, and on a `v*` tag publishes them: a tag ending in `-test` goes to TestPyPI,
any other to PyPI. Publishing uses PyPI trusted publishing (no stored token): register the project
`swan-aisql` with owner `ZhaoFuheng`, repository `SWAN-AISQL`, workflow `PythonWheels.yml` and environment
`pypi` on pypi.org (`testpypi` on test.pypi.org), and create those two environments in the repository
settings. Bumping DuckDB: pin the `duckdb/` submodule to the commit the target `duckdb` wheel reports as
`source_id`, set the same commit in both workflows and the matching `duckdb==` version in `pyproject.toml`.
