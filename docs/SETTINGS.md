# SWAN AI-SQL settings

All settings are `SET`-able; the `AI_*` environment variable in the last column seeds the default when set (the
bench harnesses use them). `SELECT name, value FROM duckdb_settings() WHERE name LIKE 'ai_%'` shows the live values.
Scope: the settings that say where a connection's requests go and how they are priced (`ai_endpoint`, `ai_model`,
`ai_api_key`, `ai_api`, `ai_reasoning_effort`, `ai_max_retries`, `ai_price_*`, `ai_ca_cert_file`, `ai_tls_verify`) and every
optimizer flag are per connection: a `SET` changes that connection only, `SET GLOBAL` the process default. The
settings that tune process-wide machinery (the request pool and its concurrency, the embeddings server, the
local cache, TypeSafe routing, logging) apply to the whole process however they are set.

**Connection and model**

| setting | values | default | what it does |
|---|---|---|---|
| `ai_endpoint` | URL (http; https in a build with OpenSSL, which the CI binaries and wheels are) | `http://localhost:4000` | OpenAI-compatible chat endpoint: litellm directly, or `http://localhost:4001` for the cache proxy (`AI_PROXY_URL`) |
| `ai_model` | model id | `gpt-5.6-luna` | chat model for every AI function (`AI_MODEL`) |
| `ai_api_key` | string | *(empty)* | bearer token sent to `ai_endpoint`; empty for the local stack (`AI_API_KEY`) |
| `ai_price_cache_write_mult` / `ai_price_cache_read_mult` | multiplier of the input rate | `1.25` / `0.1` | what the provider bills a written span and a served span at (GPT-5.6, measured). They set the prefix-cache write threshold: a span is written on first sight only when its expected later reads exceed (write − 1) / (1 − read), 0.28 by default; written tokens are priced at the write multiplier when the endpoint reports no cost (`AI_PRICE_CACHE_WRITE_MULT` / `AI_PRICE_CACHE_READ_MULT`) |
| `ai_price_input` / `ai_price_output` / `ai_price_cached` | USD per 1M tokens | `0` | price `ai_usage().cost_usd` from the token counts when the endpoint reports no cost; the litellm proxy reports it per response, so the local stack needs none (`AI_PRICE_INPUT` / `AI_PRICE_OUTPUT` / `AI_PRICE_CACHED`) |
| `ai_ca_cert_file` | path | *(empty = system store)* | CA bundle that https endpoints are verified against (`AI_CA_CERT_FILE`, then `SSL_CERT_FILE`) |
| `ai_tls_verify` | `true` / `false` | `true` | verify the server certificate of https endpoints; `false` for a self-signed local proxy (`AI_TLS_VERIFY`) |
| *(temperature)* | — | `0` | every chat request carries `temperature: 0`; after a model rejects it with HTTP 400 the process omits it (litellm's `drop_params` does the same behind the proxy) |
| `ai_reasoning_effort` | `low` / `medium` / `high` / *(empty)* | *(empty = omit)* | forwarded per request when set (`AI_REASONING_EFFORT`) |
| `ai_api` | `auto` / `responses` / `chat` | `auto` | which OpenAI API the client speaks (`AI_API`). `auto`: Chat Completions for every plain request (a little faster per call), the Responses API (`/v1/responses`, where an explicit prompt-cache breakpoint sits on the block itself, images included) only for a request that writes or reads a declared span; a span written there is read there. `responses` / `chat` force one path (chat: for endpoints without `/v1/responses`; image prefixes never cache there). Same prompts, same JSON-schema output, same answers |
| `ai_concurrency` | integer ≥ 1 | `20` | in-flight LLM requests, process-wide (`AI_MAX_CONCURRENCY`) |
| `ai_max_retries` | integer ≥ 0 | `6` | retries per request on 429/503/529 or a transient failure, exponential backoff (`AI_MAX_RETRIES`) |
| `ai_hedge` | `true` / `false` | `true` | duplicate a call still unanswered past the observed p99 latency; first answer wins (`AI_HEDGE`) |
| `ai_prefix_cache` | `true` / `false` | `true` | explicit provider prompt caching (`AI_PREFIX_CACHE`): on, every request asks for explicit cache mode and only the spans the engine declares are written (a span is declared when the engine's own estimate reaches 900 tokens, its conservative stand-in for the provider's 1,024-token minimum); off, no cache fields are sent and the provider's own default (e.g. automatic caching) applies. An endpoint that rejects the cache fields with a 400 is detected once and served plain bodies from then on. The engine never reorders a prompt; the convention is the shared input first: a pair prompt's long, repeated side before the other side, and an image before the instruction (`ai_image(url) || ' question'`), so predicates on the same image share the cached span. Written as instruction-first, the input is not shared; one breakpoint per request, after the longest prefix worth sharing: the constant text a prompt starts with (an instruction or few-shot block) when it alone clears 1,024 tokens, shared by every row of the predicate; else the leading input. ai_classify / ai_score / ai_complete have their own system prompts and schemas, so the same input under two functions is never one span |
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
| `ai_inline_ai_ctes` | `true` / `false` | `true` | inline CTEs that contain AI functions so pruning and pull-up can reach the predicate; `false` keeps them materialised and reduces the body by its readers' join keys instead (demand reduction) |
| `ai_lateral_unnest` | `true` / `false` | `true` | rewrite the planner's decorrelated `FROM s, LATERAL UNNEST(f(s.x))` into an UNNEST over s, so an AI predicate in s isn't copied to a side no join key reaches (`AI_LATERAL_UNNEST`) |
| `ai_semi_reduce` | `true` / `false` | `true` | Yannakakis semi-join reduction before any AI evaluation |
| `ai_pullup` | `true` / `false` | `true` | lift each semantic filter through every operator it commutes with (joins' preserved sides, projections, CTE main queries) to just above the highest row-removing one (`DUCKDB_SEMANTIC_PULLUP`) |
| `ai_speculative` | `true` / `false` | `true` | speculative pre-filter at a pulled-up predicate's leaf: rows whose verdict the cache already holds as false are pruned before the join, every other row passes through to the lifted predicate; no call, prediction or embedding is made at the leaf (`AI_SPECULATIVE`) |
| `ai_reorder` | `true` / `false` | `true` | ordering of AI predicates with learned selectivity (`DUCKDB_AI_REORDER`) |
| `ai_factorize` | `off` / `filters` / `all` | `all` | AI region placement: `off` places none (the join-level rewrites under `ai_join_factorize` still apply); `filters` places regions only above joins, for the AI filters and projections there; `all` also factorizes every AI call over a plain scan (`DUCKDB_AI_DEDUP`, `DUCKDB_AI_SCAN_REGION`) |
| `ai_factor_state` | `sparse` / `dense` | `sparse` | factor graph scheduler state: `sparse` keeps counters per member plus the pairs asked, enumerates pairs from per-member cursors and scores members, so memory and selection cost follow the calls made, not the domain; `dense` (the earlier scheduler, for A/B runs) keeps one byte per pair of each edge and scans the pair domain to pick work (`AI_FACTOR_STATE`) |
| `ai_factor_max_terms` | integer | `16` | a join predicate that is not a conjunction of factors is taken as the terms of its disjunctive normal form, each a conjunctive factor graph, results united; above this many terms the join runs as a region over the cross product |
| `ai_factor_pair_limit` | integer | `100000000` | largest estimated pair domain (left rows x right rows of one AI join edge) the factor graph takes under the dense scheduler, or for a cyclic edge graph (evaluated by the staged schedule, which lists each edge's live pairs); above it the join runs as a region over the cross product, which streams the pairs. The sparse scheduler takes any forest of edges |
| `ai_join_factorize` | `off` / `pushdown` / `factor` | `factor` | AI-condition joins: expand / push the region below the join / factor graph over the pair domain (`DUCKDB_AI_GROUP_JOIN`) |
| `ai_limit` | `true` / `false` | `true` | LIMIT push-down into AI evaluation (early stop) (`DUCKDB_AI_LIMIT`) |
| `ai_leaf_prior` | `mlp` / `calibrated` | `calibrated` | the P(true) a leaf is ordered by: `mlp` the selectivity model's prediction; `calibrated` the same with its logit shifted toward the leaf's pass rate observed in the query, by landed / (landed + 20) |
| `ai_leaf_lookahead` | integer ≥ 0 | `2` | a warm region keeps at most k × `ai_concurrency` calls outstanding and re-decides the rows behind an input just before it is submitted; `0` submits every pending input at once |

**Debug / experiment knobs** (`ai_debug_*`) — stable but not part of the user contract

| setting | values | default | what it does |
|---|---|---|---|
| `ai_debug_log` | csv of `region`, `dispatch`, `spec`, `yann`, `leaftexts`, `mock` | *(empty)* | stderr diagnostics per subsystem (`region` prints the leaf-region timers, `dispatch` each batch of calls the region hands to its request pool) |
| (env) `AI_MLP_FIFO` / `AI_MLP_MIN_TRAIN` / `AI_MLP_TRAIN_EVERY` | integer | `256` / `16` / 3 × `ai_concurrency` | the selectivity model's training window (most recent labelled examples), the examples a step needs, and the calls between two full-batch training steps |
| `ai_debug_prompt_variant` | `strict` / `soft` / `plain` | `strict` | `ai_filter` system prompt; `plain` sends the bare prompt (cross-engine, prompt-identical comparisons) |

Embedding-server environment (`serve/ai_embed_server.py`): `AI_EMBED_MODEL` (text model), `AI_EMBED_IMAGE_MODEL`
(CLIP model; empty = text only), `AI_EMBED_DEVICE` (default `cpu`), `AI_EMBED_PORT`, `AI_EMBED_BACKEND` (`st` / `mlx`).
`GET http://localhost:4002/` reports what it loaded.

## Environment-only controls

These have no `SET`; the client reads them once per process from the environment.

| variable | default | what it does |
|---|---|---|
| `AI_TIMEOUT_MS` | `60000` | per-request timeout of the LLM client |
| `AI_AGG_CHAR_BUDGET` | `48000` | character budget of one `ai_agg` request; longer lists are summarised in rounds |
| `AI_TURBO`, `AI_TURBO_START` | off, `4` | adaptive concurrency (turbo) and its starting level, instead of the fixed `ai_concurrency` |
| `AI_MLP_SEED`, `AI_MLP_TRAIN_EVERY`, `AI_MLP_FEATURE` | unset | selectivity model: seed, training cadence in calls, `prompt` for the single-embedding feature |
| `AI_REORDER_NO_WARMUP`, `AI_DEDUP_DEBUG` | unset | skip the cold-model warm-up; region debug output |
| `DUCKDB_AGENT_MODE` | unset | DuckDB 2.0's CLI, not the extension: set `0` when a bench runner spawns the CLI from an environment that carries an AI coding tool's marker, or the shell truncates piped output |
