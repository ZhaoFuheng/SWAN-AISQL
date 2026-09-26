//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/ai_client.hpp
//
// LLM client for AI SQL functions. Talks to an OpenAI-compatible endpoint
// (e.g. a local litellm proxy) with JSON-schema structured output.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/vector.hpp"

#include <utility>

namespace duckdb {

//! Sentinels that ai_image() wraps around an image reference embedded in a prompt. BuildRequestBody parses
//! `\x01 <ref> \x02` segments into image_url content parts (local file -> base64 data URI; http(s) -> URL);
//! the reorder/embed path routes them to the CLIP image encoder. Control chars, so they never collide with
//! real prompt text.
static constexpr char AI_IMAGE_OPEN = '\x01';
static constexpr char AI_IMAGE_CLOSE = '\x02';
//! True if `text` contains at least one ai_image() sentinel.
inline bool AITextHasImage(const string &text) {
	return text.find(AI_IMAGE_OPEN) != string::npos;
}

//! Runtime configuration for the AI functions. Read once from environment vars:
//!   AI_PROXY_URL       (default http://localhost:4000) OpenAI-compatible base URL
//!   AI_MODEL           (default gpt-5.6-luna)       model name the proxy understands
//!   AI_API_KEY         (default empty)                 bearer token for the proxy
//!   AI_TIMEOUT_MS      (default 60000)                 per-request timeout
//!   AI_MAX_CONCURRENCY (default 20)                    fixed concurrency (turbo off)
//!   AI_MAX_RETRIES     (default 6)                     per-request retries when throttled (429/503/529)
//!   AI_TURBO           (default off)                   start in adaptive-concurrency (turbo) mode
//!   AI_TURBO_START     (default 4)                     turbo starting concurrency
//!   AI_TURBO_MIN       (default 1)                     turbo concurrency floor
//!   AI_TURBO_MAX       (default 64)                    turbo ceiling (also clamped to the chunk size)
struct AIConfig {
	string base_url;
	string model;
	string api_key;
	//! Optional reasoning effort forwarded per request (low/medium/high; "" = omit). The
	//! litellm proxy can also pin it per model; this field covers direct endpoints.
	string reasoning_effort;
	int32_t timeout_ms;
	//! Fixed concurrency used when turbo mode is off.
	idx_t max_concurrency;
	//! Per-request retries on throttle (429/503/529). Applied in BOTH fixed and turbo modes.
	idx_t max_retries;
	//! Turbo (AIMD adaptive concurrency): ramp up while calls succeed, halve on 429/503/529.
	//! Bounded above by min(turbo_max, chunk size), so concurrency never exceeds the data chunk.
	bool turbo_default; // AI_TURBO: initial on/off; ai_turbo(BOOLEAN) overrides at runtime
	idx_t turbo_start;  // AI_TURBO_START: starting concurrency
	idx_t turbo_min;    // AI_TURBO_MIN: concurrency floor
	idx_t turbo_max;    // AI_TURBO_MAX: ceiling before chunk-size clamp
	//! ai_embed: a separate endpoint + model (local sentence-transformers by default). ai_embed
	//! always uses the fixed path (no turbo) and has no history, to keep CPU/memory bounded.
	string embed_url;   // AI_EMBED_URL   (default http://localhost:4002)
	string embed_model; // AI_EMBED_MODEL (default sentence-transformers/all-MiniLM-L6-v2)
	//! TypeSafe System One (Jev) as an optional backend for ai_filter (-> Noul) and ai_classify
	//! (-> Choice): typed judgments instead of a chat completion. Routed per request by the
	//! client (AIUsesTypeSafe): the function opts in via AIRequest::question, the setting enables
	//! it per function, and image-bearing prompts always stay on the chat model (Jev is text-only).
	//! The answer is folded back into the chat envelope ({"result":...}) so every parser and the
	//! reorder/selectivity machinery are untouched. Settings: ai_typesafe='filter,classify'.
	bool typesafe_filter = false;   // AI_TYPESAFE contains "filter"
	bool typesafe_classify = false; // AI_TYPESAFE contains "classify"
	string typesafe_url;            // AI_TYPESAFE_URL   (default http://localhost:4001 = the cache proxy, which
	                                //                    terminates TLS towards https://api.typesafe.ai)
	string typesafe_model;          // AI_TYPESAFE_MODEL (default jev-latest)
	string typesafe_api_key;        // TYPESAFE_API_KEY
	double typesafe_threshold = 0.5;      // Noul probability >= threshold -> true
	double typesafe_price_input = 0.042; // USD per 1M input tokens (output tokens are free)
	//! AI Region execution knobs (bridged from ai_debug_* settings): streaming sink on/off,
	//! wave-size override (0 = derived floor of 5x concurrency), debug log channels (csv).
	bool region_streaming = true;
	idx_t wave_size = 0;
	//! Waves the region may keep in flight at once (1 = fire synchronously, the original behaviour).
	//! A wave's LLM batch is only as wide as the DISTINCT PROMPTS its reps carry; where a join repeats
	//! one side's prompt across a chunk that can be a single call, leaving the pool idle behind a
	//! blocking wave. Overlapping waves refills the pool without speculating on any extra call.
	idx_t wave_overlap = 8;
	string debug_log;
	//! Local cache (layer 1): process-global response caches (chat + embed) serving repeats
	//! within and across queries in this process. In-batch single-flight dedup is unaffected.
	bool local_cache = true;
	//! Local-cache scope (mirrors lotus-ai's cache semantics): false = QUERY-level (Q1's cache
	//! never serves Q2 -- the benchmark rule, and the experiment default), true = CROSS-QUERY
	//! (process-lifetime, lotus's in-memory cache behavior when enabled).
	bool local_cache_cross_query = false;
	//! Hedged requests: a call still unanswered past the observed p99 latency (x1.2, floored at
	//! 2s, after 50 samples) fires a duplicate attempt; first response wins. Bounded extra
	//! calls (~1% + retriggers), counted honestly in llm_calls and reported as hedged_calls.
	bool hedge = true;
	//! Reuse HTTP connections across requests (keep-alive pool per endpoint URL). Off restores a
	//! fresh TCP (and TLS) handshake per call -- ~a minute of pure setup at 18k-call suite scale.
	bool http_keepalive = true;
	//! Explicit provider prefix caching (breakpoints emitted when a request carries a
	//! prompt_prefix). Default ON: the server cache canonicalizes cache-control fields out of its
	//! key, so split bodies replay against plain recorded entries. AI_PREFIX_CACHE=off opts out.
	bool prefix_cache = true;
	//! ai_debug_prompt_variant=='soft' selects the softer ai_filter system prompt (Fix 2 A/B).
	bool prompt_variant_soft = false;
	//! ai_debug_prompt_variant=='plain': send ai_filter's prompt BARE -- no system message, no
	//! json_schema response format, no prefix-cache splitting -- and read the verdict from the
	//! answer text. For cross-engine comparisons where the other system sends a bare prompt and
	//! carries its own "answer yes or no" instruction in the prompt itself (agent_bench/PLOP), so
	//! both engines key the same recorded sample. Never a default: the envelope is what makes a
	//! verdict reliable, so every other bench keeps it.
	bool prompt_variant_plain = false;
	//! ai_debug_trust_image_estimate: let speculative all-image rows use the estimate (gate +
	//! embeds) instead of the always-evaluate safety rule. Experiment knob; default off.
	bool trust_image_estimate = false;
	//! ai_debug_graph_eval=='lazy': factor-graph evaluation uses the need-driven streaming
	//! scheduler (no stage barriers) instead of the staged unary/edge passes.
	bool graph_eval_lazy = true;
	//! ai_debug_graph_eval=='lazy-adaptive': lazy scheduling with observed-selectivity priority
	//! ordering of the frontier (per-member adaptive; the graph-level analogue of per-tuple
	//! filter reordering).
	bool graph_eval_adaptive = true;
	//! ai_agg map-reduce: max characters of items packed into one LLM call before splitting.
	idx_t agg_char_budget;
	//! Fallback pricing (USD per 1M tokens) used only when the proxy does not
	//! return an x-litellm-response-cost header. 0 => cost left unestimated.
	//! price_cached applies to provider-cached input tokens (defaults to price_input).
	double price_input_per_mtok;
	double price_output_per_mtok;
	double price_cached_per_mtok;

	//! Cached singleton, initialized from the environment on first use.
	static const AIConfig &Get();
	//! Mutable view of the singleton for the extension's SET-callback bridge (settings are the
	//! runtime surface; env vars only seed the initial values above).
	static AIConfig &Mutable();
};

//! Per-query LLM usage/cost record. One entry per distinct query text.
struct AIQueryUsage {
	uint64_t query_id = 0;          // stable id assigned when the query is first seen
	string query_text;              // the SQL text (from ClientContext::GetCurrentQuery)
	uint64_t llm_calls = 0;         // actual requests sent to a model (chat + embed)
	uint64_t embed_calls = 0;       // subset of llm_calls that were ai_embed requests (chat = llm_calls - embed_calls)
	uint64_t embed_tokens = 0;      // subset of input_tokens/total_tokens from ai_embed requests
	uint64_t cache_hits = 0;        // inputs served from response cache / in-batch de-dup (no call)
	uint64_t hedged_calls = 0;      // duplicate attempts fired past the p99 hedge deadline
	uint64_t failed_calls = 0;      // requests that got no usable answer (transport error / non-2xx after retries): the AI function returned NULL
	uint64_t input_tokens = 0;      // prompt tokens
	uint64_t cached_tokens = 0;     // provider-cached prompt tokens (subset of input_tokens)
	uint64_t output_tokens = 0;     // completion tokens
	uint64_t reasoning_tokens = 0;  // reasoning tokens (subset of output_tokens)
	uint64_t total_tokens = 0;
	double cost_usd = 0.0;
};

//! Turbo mode toggle (default from AI_TURBO). When on, AIBatchComplete drives concurrency with an
//! adaptive AIMD limiter instead of the fixed AI_MAX_CONCURRENCY pool.
void AITurboSetEnabled(bool enabled);
bool AITurboEnabled();

//! Clear the local response caches (chat + embed). Returns entries cleared.
uint64_t AILocalCacheClear();

//! Snapshot all per-query usage records (ordered by query_id).
vector<AIQueryUsage> AIGetUsage();
//! Clear all usage records. Returns the number of query records cleared.
uint64_t AIResetUsage();

//===----------------------------------------------------------------------===//
// History mode: per-value conversation memory
//===----------------------------------------------------------------------===//
//! History mode toggle (default off). When on, functions record + replay history
//! keyed implicitly by the column values referenced in their prompt.
void AIHistorySetEnabled(bool enabled);
bool AIHistoryEnabled();
//! Append one (input, output) turn to the history of every key in `keys`.
void AIHistoryRecord(const vector<string> &keys, const string &input, const string &output);
//! Gather the prior turns across all `keys` (deduplicated, in order) for replay.
vector<std::pair<string, string>> AIHistoryGather(const vector<string> &keys);
//! Clear all recorded history. Returns the number of distinct keys cleared.
uint64_t AIHistoryReset();

//! A single chat-completion request.
struct AIRequest {
	//! Optional stable PREFIX for explicit provider prompt caching (GPT-5.6+): when non-empty,
	//! the user content is emitted as [prefix blocks, breakpoint block, prompt blocks] and the
	//! request carries prompt_cache_options mode=explicit. The prefix must make the total cached
	//! span (system + prefix) >= 1,024 visible tokens or the provider ignores it. Local-cache
	//! identity is prefix+prompt concatenated, so splitting never changes cache hits.
	string prompt_prefix;
	//! Declared fan-out hint for the prefix: how many calls the operator expects to share it.
	//! >= 2 makes the first dispatch the cache write (oracle policy); 0/1 = unknown, the client
	//! falls back to adaptive second-arrival-write. Never affects results, only cache economics.
	idx_t expected_reuse = 0;
	//! Dispatch-time decision, set by the client's prefix lifecycle (never by operators): emit
	//! the split content + breakpoint. False -> prefix is merged into the prompt (plain body).
	bool emit_breakpoint = false;
	//! User message content.
	string prompt;
	//! Optional system message (empty => none).
	string system_prompt;
	//! Optional JSON-schema object passed as response_format (empty => plain text output).
	string json_schema;
	//! Name for the json_schema (defaults to "response" when a schema is set).
	string schema_name;
	//! History mode: prior (user input, assistant output) turns replayed before `prompt`.
	vector<std::pair<string, string>> history;
	//! TypeSafe System One shape of this request, set by the AI function that built it (NONE for
	//! ai_complete/ai_score/ai_agg). Only consulted when the matching ai_typesafe function is on.
	enum class Question : uint8_t { NONE, NOUL, CHOICE } question = Question::NONE;
	//! System One `state` (the content judged) and `instructions` (the judgment). For a Noul the
	//! state is the whole ai_filter prompt (claim + context, byte-identical to the chat prompt);
	//! for a Choice it is the classified input, with `options` = (label, description) per category.
	string state;
	string instructions;
	vector<std::pair<string, string>> options;
};

//! True when this request will be sent to TypeSafe System One instead of the chat endpoint.
bool AIUsesTypeSafe(const AIRequest &request);

//! Result of a completion. On failure success=false and content is empty.
struct AIResult {
	string content;
	bool success = false;
};

//! Result of one embedding: the vector, or success=false on failure.
struct AIEmbedResult {
	vector<float> embedding;
	bool success = false;
};

//! Embed a batch of texts via the embeddings endpoint (sentence-transformers by default). Sends the
//! chunk's unique texts as ONE request (server-side batching) with temperature=-1 -- which marks an
//! ai_embed call and keeps its cache key distinct from any chat request. Always the fixed path (no
//! turbo) and no history. De-duplicated + query-scoped cached; output aligned 1:1 with `texts`.
vector<AIEmbedResult> AIEmbedBatch(const vector<string> &texts, const string &query_text = "");

//! Execute a batch of requests concurrently, with query-scoped de-duplication + response caching
//! (identical prompts are cached within a query, across its chunks, but never shared between queries).
//! The output vector is aligned 1:1 with the input. Token/cost usage is attributed to the
//! `query_text` (from ClientContext::GetCurrentQuery), retrievable via AIGetUsage().
//! `force_fixed` uses the bounded fixed-concurrency pool even when turbo mode is on.
//! Observed bytes-per-token for this workload/model (EMA over real text responses; 4.0 until
//! 20 samples). The prefix-cache gate divides byte lengths by this instead of assuming 4.
double AICalibratedBytesPerToken();

//! Probe the query-local response cache for `request`; true + `out` on a hit, never issues a call.
bool AICacheProbe(const AIRequest &request, const string &query_text, AIResult &out);
vector<AIResult> AIBatchComplete(const vector<AIRequest> &requests, const string &query_text = "",
                                 bool force_fixed = false);

//===----------------------------------------------------------------------===//
// Training data: (prompt, embedding, actual ai_filter result) for the selectivity MLP
//===----------------------------------------------------------------------===//
struct AITrainingExample {
	string prompt;
	vector<float> embedding;
	bool label; // the actual ai_filter boolean result
};
//! Record one training example, deduplicated by prompt (first write wins; later ones are no-ops).
void AIRecordTrainingExample(const string &prompt, const vector<float> &embedding, bool label);
//! Snapshot all recorded examples (for the ai_filter_training_data table function).
vector<AITrainingExample> AIGetTrainingData();
//! Clear all recorded examples; returns the number cleared.
uint64_t AIResetTrainingData();

//! JSON-escape a string (without surrounding quotes). Exposed for building schemas/prompts.
string AIJsonEscape(const string &input);

//! Parse a JSON object `content` and read a typed field. Return false if missing/mistyped.
bool AIParseBoolField(const string &content, const char *field, bool &out);
//! Read a yes/no verdict from a BARE answer text (no JSON envelope): the whole trimmed answer must
//! be one affirmative/negative token, so prose stays unparseable rather than being guessed at.
bool AIParseVerdictText(const string &content, bool &out);
bool AIParseDoubleField(const string &content, const char *field, double &out);
bool AIParseStringField(const string &content, const char *field, string &out);

} // namespace duckdb
