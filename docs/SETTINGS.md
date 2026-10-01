# SWAN AI-SQL settings

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
| `ai_typesafe` | csv of `filter`, `classify`, `score` | *(empty = off)* | route `ai_filter` → Noul, `ai_classify` → Choice and/or `ai_score` with integer bounds of at most 10 levels → Score to TypeSafe; other `ai_score` forms and image-bearing prompts stay on the chat model (`AI_TYPESAFE`) |
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
| `ai_speculative` | `true` / `false` | `true` | speculative pre-filter at a pulled-up predicate's leaf: rows the selectivity model expects to fail are evaluated and pruned before the join, the rest pass through to the lifted predicate (`AI_SPECULATIVE`) |
| `ai_reorder` | `true` / `false` | `true` | ordering of AI predicates with learned selectivity + speculative evaluation (`DUCKDB_AI_REORDER`) |
| `ai_factorize` | `off` / `filters` / `all` | `all` | AI region placement: none / above AI filters only / every AI call (`DUCKDB_AI_DEDUP`, `DUCKDB_AI_SCAN_REGION`) |
| `ai_join_factorize` | `off` / `pushdown` / `factor` | `factor` | AI-condition joins: expand / push the region below the join / factor graph over the pair domain (`DUCKDB_AI_GROUP_JOIN`) |
| `ai_limit` | `true` / `false` | `true` | LIMIT push-down into AI evaluation (early stop) (`DUCKDB_AI_LIMIT`) |

**Debug / experiment knobs** (`ai_debug_*`) — stable but not part of the user contract

| setting | values | default | what it does |
|---|---|---|---|
| `ai_debug_log` | csv of `region`, `dispatch`, `spec`, `yann`, `leaftexts`, `mock` | *(empty)* | stderr diagnostics per subsystem (`region` prints the leaf-region timers, `dispatch` each batch of calls the region hands to its request pool) |
| (env) `AI_MLP_FIFO` / `AI_MLP_MIN_TRAIN` / `AI_MLP_TRAIN_EVERY` | integer | `256` / `16` / 3 × `ai_concurrency` | the selectivity model's training window (most recent labelled examples), the examples a step needs, and the calls between two full-batch training steps |
| `ai_debug_prompt_variant` | `strict` / `soft` / `plain` | `strict` | `ai_filter` system prompt; `plain` sends the bare prompt (cross-engine, prompt-identical comparisons) |

Embedding-server environment (`serve/ai_embed_server.py`): `AI_EMBED_MODEL` (text model), `AI_EMBED_IMAGE_MODEL`
(CLIP model; empty = text only), `AI_EMBED_DEVICE` (default `cpu`), `AI_EMBED_PORT`, `AI_EMBED_BACKEND` (`st` / `mlx`).
`GET http://localhost:4002/` reports what it loaded.
