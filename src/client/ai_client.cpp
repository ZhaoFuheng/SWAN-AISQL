#include "ai_client.hpp"

#include "ai_prompt_cost.hpp"
#include "duckdb/common/types/blob.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "yyjson.hpp"

#include <fstream>
#include <iterator>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>

// httplib pulls in socket headers; keep it last and isolated in this TU.
#include "httplib.hpp"

using namespace duckdb_yyjson; // NOLINT

namespace duckdb {

//===--------------------------------------------------------------------===//
// Usage records: one AIQueryUsage per distinct query text, guarded by a mutex.
//===--------------------------------------------------------------------===//
static std::mutex g_usage_mutex;
static std::unordered_map<string, idx_t> g_query_index; // query_text -> index into g_queries
static vector<AIQueryUsage> g_queries;
static uint64_t g_next_query_id = 1;

// Caller must hold g_usage_mutex.
static idx_t GetOrCreateQueryIndex(const string &query_text) {
	auto it = g_query_index.find(query_text);
	if (it != g_query_index.end()) {
		return it->second;
	}
	const idx_t index = g_queries.size();
	AIQueryUsage record;
	record.query_id = g_next_query_id++;
	record.query_text = query_text;
	g_queries.push_back(std::move(record));
	g_query_index.emplace(query_text, index);
	return index;
}

//===--------------------------------------------------------------------===//
// Configuration
//===--------------------------------------------------------------------===//
static string GetEnvOr(const char *name, const string &fallback) {
	const char *value = std::getenv(name);
	return (value && value[0]) ? string(value) : fallback;
}

AIConfig &AIConfig::Mutable() {
	return const_cast<AIConfig &>(Get());
}

const AIConfig &AIConfig::Get() {
	static AIConfig config = []() {
		AIConfig c;
		c.base_url = GetEnvOr("AI_PROXY_URL", "http://localhost:4000");
		c.model = GetEnvOr("AI_MODEL", "gpt-5.6-luna");
		c.reasoning_effort = GetEnvOr("AI_REASONING_EFFORT", "");
		c.hedge = GetEnvOr("AI_HEDGE", "on") != "off" && GetEnvOr("AI_HEDGE", "on") != "0";
		c.http_keepalive = GetEnvOr("AI_HTTP_KEEPALIVE", "on") != "off";
		c.prefix_cache = GetEnvOr("AI_PREFIX_CACHE", "on") != "off";
		c.api_key = GetEnvOr("AI_API_KEY", "");
		c.timeout_ms = std::atoi(GetEnvOr("AI_TIMEOUT_MS", "60000").c_str());
		if (c.timeout_ms <= 0) {
			c.timeout_ms = 60000;
		}
		auto concurrency = std::atoi(GetEnvOr("AI_MAX_CONCURRENCY", "20").c_str());
		c.max_concurrency = concurrency > 0 ? static_cast<idx_t>(concurrency) : 20;
		auto retries = std::atoi(GetEnvOr("AI_MAX_RETRIES", "6").c_str());
		c.max_retries = retries >= 0 ? static_cast<idx_t>(retries) : 6;
		const string turbo = GetEnvOr("AI_TURBO", "");
		c.turbo_default = (turbo == "1" || turbo == "true" || turbo == "TRUE" || turbo == "on");
		auto turbo_start = std::atoi(GetEnvOr("AI_TURBO_START", "4").c_str());
		c.turbo_start = turbo_start > 0 ? static_cast<idx_t>(turbo_start) : 4;
		auto turbo_min = std::atoi(GetEnvOr("AI_TURBO_MIN", "1").c_str());
		c.turbo_min = turbo_min > 0 ? static_cast<idx_t>(turbo_min) : 1;
		auto turbo_max = std::atoi(GetEnvOr("AI_TURBO_MAX", "64").c_str());
		c.turbo_max = turbo_max > 0 ? static_cast<idx_t>(turbo_max) : 64;
		// Legacy debug env channels -> debug_log csv
		string dbg;
		if (std::getenv("AI_DEDUP_DEBUG")) {
			dbg += "region,";
		}
		if (std::getenv("DUCKDB_YANN_DEBUG")) {
			dbg += "yann,";
		}
		if (std::getenv("DUCKDB_SPEC_DEBUG")) {
			dbg += "spec,";
		}
		c.debug_log = dbg;
		const string stream = GetEnvOr("DUCKDB_AI_STREAM_DEDUP", "");
		c.region_streaming = !(stream == "off" || stream == "0");
		auto wave = std::atoi(GetEnvOr("DUCKDB_AI_STREAM_WAVE", "0").c_str());
		c.wave_size = wave > 0 ? static_cast<idx_t>(wave) : 0;
		c.embed_url = GetEnvOr("AI_EMBED_URL", "http://localhost:4002");
		c.embed_model = GetEnvOr("AI_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2");
		const string embed_images = GetEnvOr("AI_EMBED_IMAGES", "");
		auto embed_conc = std::atoi(GetEnvOr("AI_EMBED_CONCURRENCY", "4").c_str());
		c.embed_concurrency = embed_conc > 0 ? static_cast<idx_t>(embed_conc) : 4;
		auto embed_imgs = std::atoi(GetEnvOr("AI_EMBED_BATCH_IMAGES", "8").c_str());
		c.embed_batch_images = embed_imgs > 0 ? static_cast<idx_t>(embed_imgs) : 8;
		c.embed_slice = static_cast<idx_t>(std::atoi(GetEnvOr("AI_EMBED_SLICE", "100").c_str()));
		const string slice_text = GetEnvOr("AI_EMBED_SLICE_TEXT", "");
		c.embed_slice_text = !(slice_text == "0" || slice_text == "off" || slice_text == "false");
		const string warm_gate = GetEnvOr("AI_WARM_GATE", "");
		c.warm_gate = !(warm_gate == "0" || warm_gate == "off" || warm_gate == "false");
		c.embed_images = !(embed_images == "off" || embed_images == "0" || embed_images == "false");
		auto agg_budget = std::atoll(GetEnvOr("AI_AGG_CHAR_BUDGET", "48000").c_str());
		c.agg_char_budget = agg_budget > 0 ? static_cast<idx_t>(agg_budget) : 48000;
		c.price_input_per_mtok = std::atof(GetEnvOr("AI_PRICE_INPUT", "0").c_str());
		c.price_output_per_mtok = std::atof(GetEnvOr("AI_PRICE_OUTPUT", "0").c_str());
		c.price_cached_per_mtok = std::atof(GetEnvOr("AI_PRICE_CACHED", "0").c_str());
		const string typesafe = GetEnvOr("AI_TYPESAFE", "");
		c.typesafe_filter = typesafe.find("filter") != string::npos;
		c.typesafe_classify = typesafe.find("classify") != string::npos;
		c.typesafe_score = typesafe.find("score") != string::npos;
		// The client speaks plain http (no TLS in the bundled httplib): the cache proxy terminates
		// TLS and forwards /v1/systemone to https://api.typesafe.ai, as it forwards chat to litellm.
		c.typesafe_url = GetEnvOr("AI_TYPESAFE_URL", "http://localhost:4001");
		c.typesafe_model = GetEnvOr("AI_TYPESAFE_MODEL", "jev-latest");
		c.typesafe_api_key = GetEnvOr("TYPESAFE_API_KEY", "");
		c.typesafe_threshold = std::atof(GetEnvOr("AI_TYPESAFE_THRESHOLD", "0.5").c_str());
		c.typesafe_score_argmax = GetEnvOr("AI_TYPESAFE_SCORE_MODE", "expected") == "argmax";
		c.typesafe_price_input = std::atof(GetEnvOr("AI_TYPESAFE_PRICE_INPUT", "0.042").c_str());
		return c;
	}();
	return config;
}

bool AIUsesTypeSafe(const AIRequest &request) {
	const auto &config = AIConfig::Get();
	// Jev is text-only: an image sentinel ANYWHERE in what would be sent (a join may carry the
	// image side as the declared prefix, not the prompt) keeps the request on the chat model.
	const bool has_image =
	    AITextHasImage(request.prompt_prefix) || AITextHasImage(request.prompt) || AITextHasImage(request.state);
	switch (request.question) {
	case AIRequest::Question::NOUL:
		return config.typesafe_filter && !has_image;
	case AIRequest::Question::CHOICE:
		return config.typesafe_classify && !request.options.empty() && !has_image;
	case AIRequest::Question::SCORE:
		return config.typesafe_score && request.options.size() >= 2 && request.options.size() <= 10 && !has_image;
	default:
		return false;
	}
}

//! Turbo toggle: -1 = follow AI_TURBO default, 0 = forced off, 1 = forced on (via ai_turbo()).
static std::atomic<int8_t> g_turbo_state {-1};

void AITurboSetEnabled(bool enabled) {
	g_turbo_state.store(enabled ? 1 : 0);
}

bool AITurboEnabled() {
	const int8_t state = g_turbo_state.load();
	if (state < 0) {
		return AIConfig::Get().turbo_default;
	}
	return state == 1;
}

//===--------------------------------------------------------------------===//
// JSON helpers
//===--------------------------------------------------------------------===//
string AIJsonEscape(const string &input) {
	string out;
	out.reserve(input.size() + 8);
	for (unsigned char c : input) {
		switch (c) {
		case '"':
			out += "\\\"";
			break;
		case '\\':
			out += "\\\\";
			break;
		case '\b':
			out += "\\b";
			break;
		case '\f':
			out += "\\f";
			break;
		case '\n':
			out += "\\n";
			break;
		case '\r':
			out += "\\r";
			break;
		case '\t':
			out += "\\t";
			break;
		default:
			if (c < 0x20) {
				char buf[8];
				snprintf(buf, sizeof(buf), "\\u%04x", c);
				out += buf;
			} else {
				out += static_cast<char>(c);
			}
		}
	}
	return out;
}

//! Extract choices[0].message.content from an OpenAI-compatible chat response.
static bool ParseChatContent(const string &text, string &out) {
	yyjson_doc *doc = yyjson_read(text.c_str(), text.size(), 0);
	if (!doc) {
		return false;
	}
	bool ok = false;
	yyjson_val *root = yyjson_doc_get_root(doc);
	if (root && yyjson_is_obj(root)) {
		yyjson_val *choices = yyjson_obj_get(root, "choices");
		if (choices && yyjson_is_arr(choices)) {
			yyjson_val *first = yyjson_arr_get(choices, 0);
			if (first) {
				yyjson_val *message = yyjson_obj_get(first, "message");
				if (message) {
					yyjson_val *content = yyjson_obj_get(message, "content");
					if (content && yyjson_is_str(content)) {
						out.assign(yyjson_get_str(content), yyjson_get_len(content));
						ok = true;
					}
				}
			}
		}
	}
	yyjson_doc_free(doc);
	return ok;
}

bool AIParseBoolField(const string &content, const char *field, bool &out) {
	yyjson_doc *doc = yyjson_read(content.c_str(), content.size(), 0);
	if (!doc) {
		return false;
	}
	bool ok = false;
	yyjson_val *root = yyjson_doc_get_root(doc);
	if (root && yyjson_is_obj(root)) {
		yyjson_val *v = yyjson_obj_get(root, field);
		if (v && yyjson_is_bool(v)) {
			out = yyjson_get_bool(v);
			ok = true;
		} else if (v && yyjson_is_str(v)) {
			string s(yyjson_get_str(v), yyjson_get_len(v));
			std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return std::tolower(ch); });
			out = (s == "true" || s == "yes" || s == "1");
			ok = true;
		}
	}
	yyjson_doc_free(doc);
	return ok;
}

bool AIParseVerdictText(const string &content, bool &out) {
	// Strip surrounding whitespace and the punctuation/markdown a model puts around a one-word
	// answer ("**Yes**", "no."). What remains must be the WHOLE answer: a verdict mined out of a
	// sentence would be a guess, and a wrong guess is indistinguishable from a real verdict.
	const string trim_set = " \t\r\n.!*\"'`";
	const auto b = content.find_first_not_of(trim_set);
	if (b == string::npos) {
		return false;
	}
	const auto e = content.find_last_not_of(trim_set);
	string s = content.substr(b, e - b + 1);
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return std::tolower(ch); });
	if (s == "yes" || s == "true" || s == "y" || s == "1") {
		out = true;
		return true;
	}
	if (s == "no" || s == "false" || s == "n" || s == "0") {
		out = false;
		return true;
	}
	return false;
}

bool AIParseDoubleField(const string &content, const char *field, double &out) {
	yyjson_doc *doc = yyjson_read(content.c_str(), content.size(), 0);
	if (!doc) {
		return false;
	}
	bool ok = false;
	yyjson_val *root = yyjson_doc_get_root(doc);
	if (root && yyjson_is_obj(root)) {
		yyjson_val *v = yyjson_obj_get(root, field);
		if (v && yyjson_is_num(v)) {
			out = yyjson_get_num(v);
			ok = true;
		} else if (v && yyjson_is_str(v)) {
			try {
				out = std::stod(string(yyjson_get_str(v), yyjson_get_len(v)));
				ok = true;
			} catch (...) {
				ok = false;
			}
		}
	}
	yyjson_doc_free(doc);
	return ok;
}

bool AIParseStringField(const string &content, const char *field, string &out) {
	yyjson_doc *doc = yyjson_read(content.c_str(), content.size(), 0);
	if (!doc) {
		return false;
	}
	bool ok = false;
	yyjson_val *root = yyjson_doc_get_root(doc);
	if (root && yyjson_is_obj(root)) {
		yyjson_val *v = yyjson_obj_get(root, field);
		if (v && yyjson_is_str(v)) {
			out.assign(yyjson_get_str(v), yyjson_get_len(v));
			ok = true;
		}
	}
	yyjson_doc_free(doc);
	return ok;
}

//! Parse the `usage` block + litellm cost header of a successful response and
//! fold the token counts / dollar cost into the record for `query_index`.
//===--------------------------------------------------------------------===//
// Self-calibrating token estimation: every real response reports its true prompt_tokens, so the
// observed bytes-per-token ratio for THIS workload and model is free. An EMA over text-only
// calls beats any vendored tokenizer here: it reflects our schemas, prompt idioms and language
// mix, tracks model/tokenizer changes automatically, and adds zero dependencies. Consumers that
// need absolute-ish counts (the prefix-cache gate) divide bytes by this ratio; the relative-cost
// consumers (DP ordering) keep the plain bytes/4 stand-in, where only monotonicity matters.
//===--------------------------------------------------------------------===//
static std::mutex g_tokratio_mutex;
static double g_tokratio_ema = 4.0;
static idx_t g_tokratio_samples = 0;

static void ObserveTokenRatio(const string &body, uint64_t prompt_tokens) {
	// text calls only (image base64 dominates bytes with unrelated token economics), and only
	// substantial ones (the mock's fixed 8-token usage would poison the ratio)
	if (prompt_tokens < 50 || body.find("image_url") != string::npos) {
		return;
	}
	const double ratio = static_cast<double>(body.size()) / static_cast<double>(prompt_tokens);
	if (ratio < 1.0 || ratio > 16.0) {
		return; // implausible: malformed usage or exotic payload
	}
	std::lock_guard<std::mutex> lock(g_tokratio_mutex);
	g_tokratio_ema = g_tokratio_ema * 0.9375 + ratio * 0.0625; // alpha = 1/16
	g_tokratio_samples++;
}

double AICalibratedBytesPerToken() {
	std::lock_guard<std::mutex> lock(g_tokratio_mutex);
	return g_tokratio_samples >= 20 ? g_tokratio_ema : 4.0;
}

//! `typesafe`: the body is a System One response (usage.input_tokens/output_tokens, priced per
//! input token from typesafe_price_input since no proxy cost header exists for it).
static void AccountUsage(idx_t query_index, const string &body, const string &cost_header, const AIConfig &config,
                         bool typesafe = false) {
	uint64_t input_tokens = 0, output_tokens = 0, cached_tokens = 0, reasoning_tokens = 0, total_tokens = 0;
	yyjson_doc *doc = yyjson_read(body.c_str(), body.size(), 0);
	if (doc) {
		yyjson_val *root = yyjson_doc_get_root(doc);
		yyjson_val *usage = root && yyjson_is_obj(root) ? yyjson_obj_get(root, "usage") : nullptr;
		if (usage && yyjson_is_obj(usage)) {
			yyjson_val *v;
			if ((v = yyjson_obj_get(usage, typesafe ? "input_tokens" : "prompt_tokens")) && yyjson_is_num(v)) {
				input_tokens = static_cast<uint64_t>(yyjson_get_num(v));
			}
			if ((v = yyjson_obj_get(usage, typesafe ? "output_tokens" : "completion_tokens")) && yyjson_is_num(v)) {
				output_tokens = static_cast<uint64_t>(yyjson_get_num(v));
			}
			if ((v = yyjson_obj_get(usage, "total_tokens")) && yyjson_is_num(v)) {
				total_tokens = static_cast<uint64_t>(yyjson_get_num(v));
			}
			yyjson_val *prompt_details = yyjson_obj_get(usage, "prompt_tokens_details");
			if (prompt_details && yyjson_is_obj(prompt_details) &&
			    (v = yyjson_obj_get(prompt_details, "cached_tokens")) && yyjson_is_num(v)) {
				cached_tokens = static_cast<uint64_t>(yyjson_get_num(v));
			}
			yyjson_val *completion_details = yyjson_obj_get(usage, "completion_tokens_details");
			if (completion_details && yyjson_is_obj(completion_details) &&
			    (v = yyjson_obj_get(completion_details, "reasoning_tokens")) && yyjson_is_num(v)) {
				reasoning_tokens = static_cast<uint64_t>(yyjson_get_num(v));
			}
		}
		yyjson_doc_free(doc);
	}
	if (total_tokens == 0) {
		total_tokens = input_tokens + output_tokens;
	}

	// Prefer the proxy-computed cost; otherwise estimate from the fallback price table.
	double cost = 0.0;
	if (typesafe) {
		cost = (static_cast<double>(input_tokens) / 1e6) * config.typesafe_price_input;
	} else if (!cost_header.empty()) {
		try {
			cost = std::stod(cost_header);
		} catch (...) {
			cost = 0.0;
		}
	}
	if (cost == 0.0 &&
	    (config.price_input_per_mtok > 0 || config.price_output_per_mtok > 0 || config.price_cached_per_mtok > 0)) {
		const double cached_price =
		    config.price_cached_per_mtok > 0 ? config.price_cached_per_mtok : config.price_input_per_mtok;
		const uint64_t uncached_input = input_tokens > cached_tokens ? input_tokens - cached_tokens : 0;
		cost = (static_cast<double>(uncached_input) / 1e6) * config.price_input_per_mtok +
		       (static_cast<double>(cached_tokens) / 1e6) * cached_price +
		       (static_cast<double>(output_tokens) / 1e6) * config.price_output_per_mtok;
	}

	ObserveTokenRatio(body, input_tokens);
	std::lock_guard<std::mutex> lock(g_usage_mutex);
	auto &record = g_queries[query_index];
	record.llm_calls += 1;
	record.input_tokens += input_tokens;
	record.cached_tokens += cached_tokens;
	record.output_tokens += output_tokens;
	record.reasoning_tokens += reasoning_tokens;
	record.total_tokens += total_tokens;
	record.cost_usd += cost;
}

//===--------------------------------------------------------------------===//
// HTTP request
//===--------------------------------------------------------------------===//
// Resolve an ai_image() reference to a value usable in an OpenAI image_url: an http(s) URL is passed
// through; a local file path is read and base64-encoded into a data: URI (mime inferred from extension).
static string BuildImageUrl(const string &ref) {
	if (ref.rfind("http://", 0) == 0 || ref.rfind("https://", 0) == 0 || ref.rfind("data:", 0) == 0) {
		return ref;
	}
	std::ifstream f(ref, std::ios::binary);
	if (!f) {
		return ref; // unreadable: pass the raw ref (upstream will error) rather than crash
	}
	const string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	string mime = "image/png";
	const auto dot = ref.find_last_of('.');
	if (dot != string::npos) {
		string ext = ref.substr(dot + 1);
		for (auto &c : ext) {
			c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
		}
		if (ext == "jpg" || ext == "jpeg") {
			mime = "image/jpeg";
		} else if (ext == "gif") {
			mime = "image/gif";
		} else if (ext == "webp") {
			mime = "image/webp";
		}
	}
	const string b64 = Blob::ToBase64(string_t(bytes.data(), static_cast<uint32_t>(bytes.size())));
	return "data:" + mime + ";base64," + b64;
}

// Build the JSON value for a user message's "content": a plain quoted string when there is no image, or an
// array of text / image_url parts when the prompt contains ai_image() sentinels (\x01 ref \x02).
static void EmitContentBlocks(string &arr, bool &first, const string &prompt);

static string BuildUserContent(const string &prompt) {
	if (!AITextHasImage(prompt)) {
		return "\"" + AIJsonEscape(prompt) + "\"";
	}
	string arr = "[";
	bool first = true;
	EmitContentBlocks(arr, first, prompt);
	arr += "]";
	return arr;
}

//! Prefixed form for explicit provider prompt caching: [prefix blocks][breakpoint block][suffix
//! blocks]. The breakpoint rides on a dedicated newline text block so it works uniformly whether
//! the prefix ends in text or an image.
static string BuildUserContentPrefixed(const string &prefix, const string &suffix) {
	string arr = "[";
	bool first = true;
	if (!AITextHasImage(prefix)) {
		// text-only prefix: the breakpoint rides on the prefix block itself, so the visible text
		// is exactly prefix+suffix (identical to the unsplit prompt -- mock/replay seed identity)
		arr += "{\"type\":\"text\",\"text\":\"" + AIJsonEscape(prefix) +
		       "\",\"prompt_cache_breakpoint\":{\"mode\":\"explicit\"}}";
		first = false;
	} else {
		// image-ending prefixes need a text carrier block for the breakpoint
		EmitContentBlocks(arr, first, prefix);
		arr += (first ? "" : ",");
		arr += "{\"type\":\"text\",\"text\":\"\\n\",\"prompt_cache_breakpoint\":{\"mode\":\"explicit\"}}";
		first = false;
	}
	EmitContentBlocks(arr, first, suffix);
	arr += "]";
	return arr;
}

static void EmitContentBlocks(string &arr, bool &first, const string &prompt) {
	auto emit_text = [&](const string &t) {
		if (t.empty()) {
			return;
		}
		arr += (first ? "" : ",");
		arr += "{\"type\":\"text\",\"text\":\"" + AIJsonEscape(t) + "\"}";
		first = false;
	};
	auto emit_image = [&](const string &ref) {
		arr += (first ? "" : ",");
		arr += "{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + AIJsonEscape(BuildImageUrl(ref)) + "\"}}";
		first = false;
	};
	size_t i = 0;
	while (i < prompt.size()) {
		const size_t open = prompt.find(AI_IMAGE_OPEN, i);
		if (open == string::npos) {
			emit_text(prompt.substr(i));
			break;
		}
		emit_text(prompt.substr(i, open - i));
		const size_t close = prompt.find(AI_IMAGE_CLOSE, open + 1);
		if (close == string::npos) {
			emit_text(prompt.substr(open + 1)); // malformed (no close): keep the rest as text
			break;
		}
		emit_image(prompt.substr(open + 1, close - open - 1));
		i = close + 1;
	}
}

static string BuildRequestBody(const AIConfig &config, const AIRequest &request) {
	// temperature=0 for determinism; proxy drop_params handles models that reject it.
	string body = "{\"model\":\"" + AIJsonEscape(config.model) + "\",\"temperature\":0,";
	if (!config.reasoning_effort.empty()) {
		body += "\"reasoning_effort\":\"" + AIJsonEscape(config.reasoning_effort) + "\",";
	}
	body += "\"messages\":[";
	bool first = true;
	if (!request.system_prompt.empty()) {
		body += "{\"role\":\"system\",\"content\":\"" + AIJsonEscape(request.system_prompt) + "\"}";
		first = false;
	}
	// History mode: replay prior turns as user/assistant message pairs.
	for (const auto &turn : request.history) {
		if (!first) {
			body += ",";
		}
		body += "{\"role\":\"user\",\"content\":\"" + AIJsonEscape(turn.first) + "\"},";
		body += "{\"role\":\"assistant\",\"content\":\"" + AIJsonEscape(turn.second) + "\"}";
		first = false;
	}
	if (!first) {
		body += ",";
	}
	if (request.emit_breakpoint && !request.prompt_prefix.empty()) {
		body += "{\"role\":\"user\",\"content\":" +
		        BuildUserContentPrefixed(request.prompt_prefix, request.prompt) + "}";
	} else if (!request.prompt_prefix.empty()) {
		body += "{\"role\":\"user\",\"content\":" + BuildUserContent(request.prompt_prefix + request.prompt) + "}";
	} else {
		body += "{\"role\":\"user\",\"content\":" + BuildUserContent(request.prompt) + "}";
	}
	body += "]";
	if (request.emit_breakpoint && !request.prompt_prefix.empty()) {
		body += ",\"prompt_cache_options\":{\"mode\":\"explicit\"}";
	}
	if (!request.json_schema.empty()) {
		const string name = request.schema_name.empty() ? "response" : request.schema_name;
		body += ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"" + AIJsonEscape(name) +
		        "\",\"strict\":true,\"schema\":" + request.json_schema + "}}";
	}
	body += "}";
	return body;
}

//===--------------------------------------------------------------------===//
// TypeSafe System One: POST /v1/systemone {state, model, questions:{q:{type, instructions,
// criteria}}} -> {answers:{q:{noul|choice}}, usage:{input_tokens, output_tokens}}. One question
// per request keeps the call/cache/dedup accounting identical to the chat path (one prompt = one
// call); the answer is re-encoded as the chat envelope's {"result": ...} content.
//===--------------------------------------------------------------------===//
static string BuildSystemOneBody(const AIConfig &config, const AIRequest &request) {
	// A Noul judges the whole ai_filter prompt (any declared prefix included, exactly as the chat
	// path would send it); a Choice judges the classified input the function set as `state`.
	const string state = request.question == AIRequest::Question::NOUL ? request.prompt_prefix + request.prompt
	                                                                     : request.state;
	string body = "{\"model\":\"" + AIJsonEscape(config.typesafe_model) + "\",\"state\":\"" + AIJsonEscape(state) +
	              "\",\"questions\":{\"q\":{";
	if (request.question == AIRequest::Question::NOUL) {
		body += "\"type\":\"noul\",\"instructions\":\"" + AIJsonEscape(request.instructions) +
		        "\",\"criteria\":{\"true\":\"The claim holds for the context.\","
		        "\"false\":\"The claim does not hold for the context.\"}";
	} else if (request.question == AIRequest::Question::SCORE) {
		// A Score takes its levels as an ORDERED array, low to high; the answer is the expected level.
		body += "\"type\":\"score\",\"instructions\":\"" + AIJsonEscape(request.instructions) + "\",\"criteria\":[";
		for (idx_t i = 0; i < request.options.size(); i++) {
			body += (i ? ",\"" : "\"") + AIJsonEscape(request.options[i].second) + "\"";
		}
		body += "]";
	} else {
		body += "\"type\":\"choice\",\"instructions\":\"" + AIJsonEscape(request.instructions) + "\",\"criteria\":{";
		for (idx_t i = 0; i < request.options.size(); i++) {
			if (i) {
				body += ",";
			}
			body += "\"" + AIJsonEscape(request.options[i].first) + "\":";
			body += request.options[i].second.empty() ? string("null")
			                                          : "\"" + AIJsonEscape(request.options[i].second) + "\"";
		}
		body += "}";
	}
	body += "}}}";
	return body;
}

//! Fold the System One answer into chat-envelope content: {"result":true|false} for a Noul
//! (probability >= threshold), {"result":"<choice>"} for a Choice, {"result":<number>} for a Score
//! (the probability-weighted level, mapped back onto the function's scale). False on a malformed body.
static bool ParseSystemOneAnswer(const string &text, const AIConfig &config, const AIRequest &request, string &out) {
	yyjson_doc *doc = yyjson_read(text.c_str(), text.size(), 0);
	if (!doc) {
		return false;
	}
	bool ok = false;
	yyjson_val *root = yyjson_doc_get_root(doc);
	yyjson_val *answers = root && yyjson_is_obj(root) ? yyjson_obj_get(root, "answers") : nullptr;
	yyjson_val *q = answers && yyjson_is_obj(answers) ? yyjson_obj_get(answers, "q") : nullptr;
	if (q && yyjson_is_obj(q)) {
		if (request.question == AIRequest::Question::NOUL) {
			yyjson_val *p = yyjson_obj_get(q, "noul");
			if (p && yyjson_is_num(p)) {
				out = yyjson_get_num(p) >= config.typesafe_threshold ? "{\"result\":true}" : "{\"result\":false}";
				ok = true;
			}
		} else if (request.question == AIRequest::Question::SCORE) {
			// `score` is the probability-weighted level; argmax mode takes the most probable level from
			// `probabilities` instead. Level 0 is the function's `lo`; the scalar/leaf then clamp and round
			// exactly as for a chat answer.
			yyjson_val *s = yyjson_obj_get(q, "score");
			double level = s && yyjson_is_num(s) ? yyjson_get_num(s) : -1;
			yyjson_val *probs = yyjson_obj_get(q, "probabilities");
			if (config.typesafe_score_argmax && probs && yyjson_is_obj(probs)) {
				double best = -1;
				size_t pi, pmax;
				yyjson_val *pk, *pv;
				yyjson_obj_foreach(probs, pi, pmax, pk, pv) {
					if (yyjson_is_num(pv) && yyjson_get_num(pv) > best) {
						best = yyjson_get_num(pv);
						level = std::atof(yyjson_get_str(pk));
					}
				}
			}
			if (level >= 0) {
				out = "{\"result\":" + std::to_string(request.score_lo + level) + "}";
				ok = true;
			}
		} else {
			yyjson_val *c = yyjson_obj_get(q, "choice");
			if (c && yyjson_is_str(c)) {
				out = "{\"result\":\"" + AIJsonEscape(string(yyjson_get_str(c), yyjson_get_len(c))) + "\"}";
				ok = true;
			}
		}
	}
	yyjson_doc_free(doc);
	return ok;
}

//! Outcome of one HTTP attempt, used by turbo mode to steer concurrency.
enum class AICallOutcome : uint8_t {
	OK,        // 2xx with a parseable body
	THROTTLED, // 429 / 503 / 529 -> back off, retry, and shrink concurrency
	TRANSIENT, // network error / 5xx / unparseable body -> back off and retry (not a rate signal)
	ERROR      // permanent failure (4xx other than throttle) -> not retried
};

//! Read a Retry-After / retry-after-ms hint (in ms) from a throttle response; -1 if absent.
static int32_t ParseRetryAfterMs(const duckdb_httplib::Response &response) {
	const string ms = response.get_header_value("retry-after-ms");
	if (!ms.empty()) {
		try {
			return std::stoi(ms);
		} catch (...) {
		}
	}
	const string secs = response.get_header_value("retry-after");
	if (!secs.empty()) {
		try {
			return std::stoi(secs) * 1000;
		} catch (...) {
		}
	}
	return -1;
}

//===--------------------------------------------------------------------===//
// Global chat gate: a hard process-wide cap of ai_concurrency in-flight CHAT requests, acquired
// around every HTTP attempt. Batch pools, streaming graph units and hedge duplicates all draw
// from the same permits, so concurrent operators (or hedging) can never stack past the cap.
//===--------------------------------------------------------------------===//
static std::mutex g_gate_mutex;
static std::condition_variable g_gate_cv;
static idx_t g_gate_in_flight = 0;

class ChatGatePermit {
public:
	ChatGatePermit() {
		const idx_t cap = MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1);
		std::unique_lock<std::mutex> lock(g_gate_mutex);
		g_gate_cv.wait(lock, [&]() { return g_gate_in_flight < cap; });
		g_gate_in_flight++;
	}
	~ChatGatePermit() {
		{
			std::lock_guard<std::mutex> lock(g_gate_mutex);
			g_gate_in_flight--;
		}
		g_gate_cv.notify_one();
	}
};

//===--------------------------------------------------------------------===//
// HTTP connection pool: keep-alive clients reused across requests, keyed by endpoint URL.
// httplib::Client is single-threaded, so checkout grants exclusive ownership; checkin returns a
// connection that completed an HTTP exchange (any status -- 429 still means a healthy socket).
// A reused socket the peer closed while idle surfaces as a failed Post: the caller retries once
// on a fresh connection before reporting TRANSIENT, so failure semantics match the old
// client-per-request behavior exactly.
//===--------------------------------------------------------------------===//
static std::mutex g_conn_pool_mutex;
static std::unordered_map<string, vector<duckdb::unique_ptr<duckdb_httplib::Client>>> g_conn_pool;
static constexpr idx_t CONN_POOL_MAX_IDLE = 64; // per URL; > concurrency + hedges + embed callers

static duckdb::unique_ptr<duckdb_httplib::Client> NewConn(const string &url, const AIConfig &config) {
	auto client = duckdb::make_uniq<duckdb_httplib::Client>(url);
	client->set_keep_alive(config.http_keepalive);
	return client;
}

//! RAII lease on a pooled connection. `reused` tells the caller whether a failed Post may just be
//! a stale keep-alive socket (retry on a fresh one) or a real transport error.
struct ConnLease {
	ConnLease(const string &url_p, const AIConfig &config_p) : url(url_p), config(config_p) {
		if (config.http_keepalive) {
			std::lock_guard<std::mutex> lock(g_conn_pool_mutex);
			auto it = g_conn_pool.find(url);
			if (it != g_conn_pool.end() && !it->second.empty()) {
				client = std::move(it->second.back());
				it->second.pop_back();
				reused = true;
			}
		}
		if (!client) {
			client = NewConn(url, config);
		}
		const time_t sec = config.timeout_ms / 1000;
		const time_t usec = (config.timeout_ms % 1000) * 1000;
		client->set_connection_timeout(sec, usec);
		client->set_read_timeout(sec, usec);
		client->set_write_timeout(sec, usec);
	}
	//! Replace a stale reused socket with a fresh connection (in-place single retry).
	void Refresh() {
		client = NewConn(url, config);
		const time_t sec = config.timeout_ms / 1000;
		const time_t usec = (config.timeout_ms % 1000) * 1000;
		client->set_connection_timeout(sec, usec);
		client->set_read_timeout(sec, usec);
		client->set_write_timeout(sec, usec);
		reused = false;
	}
	~ConnLease() {
		if (!healthy || !config.http_keepalive || !client) {
			return; // drop: broken socket, or pooling disabled
		}
		std::lock_guard<std::mutex> lock(g_conn_pool_mutex);
		auto &idle = g_conn_pool[url];
		if (idle.size() < CONN_POOL_MAX_IDLE) {
			idle.push_back(std::move(client));
		}
	}
	string url;
	const AIConfig &config;
	duckdb::unique_ptr<duckdb_httplib::Client> client;
	bool reused = false;
	bool healthy = false; // set by the caller once an HTTP exchange completed on this socket
};

//! `error` describes a failed attempt (transport error, HTTP status + body head, unparseable body) for
//! the diagnostic the caller prints once the request is given up on.
static AIResult DoSingleRequest(const AIConfig &config, const AIRequest &request, idx_t query_index,
                                AICallOutcome &outcome, int32_t &retry_after_ms, string &error) {
	ChatGatePermit gate_permit; // global in-flight cap, held for this attempt only
	outcome = AICallOutcome::ERROR;
	retry_after_ms = -1;
	error.clear();
	AIResult result;
	if (request.prompt.empty() && request.system_prompt.empty()) {
		outcome = AICallOutcome::OK; // nothing to send: benign, not a throttle
		return result;
	}
	const bool typesafe = AIUsesTypeSafe(request);
	const string &endpoint = typesafe ? config.typesafe_url : config.base_url;
	duckdb::unique_ptr<ConnLease> lease_holder;
	try {
		lease_holder = duckdb::make_uniq<ConnLease>(endpoint, config);
	} catch (const std::exception &e) {
		// e.g. an https URL in a build without TLS support: a permanent, reported failure -- never
		// an uncaught exception on a worker thread.
		error = string("cannot open a connection: ") + e.what() + " (route it through an http proxy, e.g. serve/ai_cache_server.py)";
		return result;
	}
	ConnLease &lease = *lease_holder;

	duckdb_httplib::Headers headers;
	const string &api_key = typesafe ? config.typesafe_api_key : config.api_key;
	if (!api_key.empty()) {
		headers.emplace("Authorization", "Bearer " + api_key);
	}
	// Wall-clock start (ms since epoch). A caching proxy uses it to reproduce the original latency
	// from the client's point of view -- replying at start+latency rather than adding its own
	// receive/queue overhead. Carried as a header so it never affects the request-body cache key.
	const auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                          std::chrono::system_clock::now().time_since_epoch())
	                          .count();
	headers.emplace("X-Request-Start-Ms", std::to_string(start_ms));
	const string body = typesafe ? BuildSystemOneBody(config, request) : BuildRequestBody(config, request);
	const char *path = typesafe ? "/v1/systemone" : "/v1/chat/completions";
	auto response = lease.client->Post(path, headers, body, "application/json");
	if (!response && lease.reused) {
		// A pooled socket the peer closed while idle: retry once on a fresh connection so the
		// failure semantics match the old client-per-request behavior.
		lease.Refresh();
		response = lease.client->Post(path, headers, body, "application/json");
	}
	if (!response) {
		outcome = AICallOutcome::TRANSIENT; // network error (timeout / reset): retry
		error = "no response (" + duckdb_httplib::to_string(response.error()) + ")";
		return result;
	}
	lease.healthy = true; // full HTTP exchange completed: socket reusable regardless of status
	auto http_error = [&](const char *what) {
		error = string(what) + " HTTP " + std::to_string(response->status) + ": " + response->body.substr(0, 200);
	};
	if (response->status == 429 || response->status == 503 || response->status == 529) {
		outcome = AICallOutcome::THROTTLED;
		retry_after_ms = ParseRetryAfterMs(*response);
		http_error("throttled,");
		return result;
	}
	if (response->status >= 500) {
		outcome = AICallOutcome::TRANSIENT; // upstream 5xx (500/502/504): retry
		http_error("upstream error,");
		return result;
	}
	if (response->status < 200 || response->status >= 300) {
		http_error("rejected,");
		return result; // other 4xx: permanent, do not retry
	}
	string content;
	const bool parsed = typesafe ? ParseSystemOneAnswer(response->body, config, request, content)
	                             : ParseChatContent(response->body, content);
	if (!parsed) {
		outcome = AICallOutcome::TRANSIENT; // 2xx but unparseable (e.g. truncated under load): retry
		error = "unparseable response body: " + response->body.substr(0, 200);
		return result;
	}
	AccountUsage(query_index, response->body, response->get_header_value("x-litellm-response-cost"), config,
	             typesafe);
	result.content = std::move(content);
	result.success = true;
	outcome = AICallOutcome::OK;
	return result;
}

//===--------------------------------------------------------------------===//
// Batch execution: de-dup + cache + bounded concurrency
//===--------------------------------------------------------------------===//
static std::mutex g_cache_mutex;
static std::unordered_map<string, AIResult> g_cache;

//! A prompt this process has already SENT but not yet answered. The cache alone cannot dedup
//! these: it is only written when a response lands, so two batches that start the same prompt
//! concurrently both miss and both pay. That is routine once a CTE carrying an AI filter is
//! inlined -- each reference becomes its own copy of the same scan, running in parallel over the
//! same rows (agent_bench Q23: 91 distinct prompts, 131 calls across 3 references).
struct AIInFlight {
	std::mutex m;
	std::condition_variable cv;
	bool done = false;
	AIResult result;
};
//! key -> the request in flight for it. Guarded by g_cache_mutex; entries live only between
//! dispatch and publication, and waiters hold a shared_ptr so erasing never strands them.
static std::unordered_map<string, shared_ptr<AIInFlight>> g_inflight;

//! Publishes an owned in-flight entry and retires it from the registry. RAII so an exception on
//! the dispatch path cannot leave a waiter blocked forever -- it then publishes the default
//! (failed) result, which is what an un-cached failure means to the caller anyway.
struct AIInFlightPublisher {
	vector<shared_ptr<AIInFlight>> entries;
	vector<string> keys;

	void Add(shared_ptr<AIInFlight> e, string key) {
		entries.push_back(std::move(e));
		keys.push_back(std::move(key));
	}
	//! Hand each waiter the owner's result. Called before the registry entries are dropped.
	void Publish(const vector<AIResult> &results) {
		for (idx_t u = 0; u < entries.size() && u < results.size(); u++) {
			{
				std::lock_guard<std::mutex> lock(entries[u]->m);
				entries[u]->result = results[u];
				entries[u]->done = true;
			}
			entries[u]->cv.notify_all();
		}
	}
	~AIInFlightPublisher() {
		for (auto &e : entries) {
			if (!e->done) {
				{
					std::lock_guard<std::mutex> lock(e->m);
					e->done = true;
				}
				e->cv.notify_all();
			}
		}
		std::lock_guard<std::mutex> lock(g_cache_mutex);
		for (auto &k : keys) {
			g_inflight.erase(k);
		}
	}
};

//! Cache-key scope prefix: per-query index (default; Q1 never serves Q2) or a fixed global
//! prefix when ai_local_cache_scope='cross_query' (lotus-style process-lifetime reuse).
static string CacheScopePrefix(const AIConfig &config, idx_t query_index) {
	if (config.local_cache_cross_query) {
		return string("g\x1f");
	}
	return std::to_string(query_index) + "\x1f";
}

static string RequestSignature(const AIConfig &config, const AIRequest &request) {
	string sig = config.model + "\x1f" + config.reasoning_effort + "\x1f" + request.system_prompt + "\x1f" +
	             request.json_schema + "\x1f" + request.prompt_prefix + request.prompt;
	if (AIUsesTypeSafe(request)) {
		// A System One answer is a different sample from a chat answer to the same prompt.
		sig += "\x1e" "typesafe\x1f" + config.typesafe_model + "\x1f" + request.instructions;
		for (const auto &opt : request.options) {
			sig += "\x1f" + opt.first + "=" + opt.second;
		}
	}
	// History changes the response, so it must be part of the cache identity.
	for (const auto &turn : request.history) {
		sig += "\x1e" + turn.first + "\x1f" + turn.second;
	}
	return sig;
}

//! Query-local cache probe: serve `request` from this query's response cache WITHOUT issuing a
//! call on a miss. Backs cache-only speculative pruning (LIMIT stand-down): pruning is free when
//! the answer is already known, and a miss must never spend a call the k-bounded evaluation above
//! may not need. A hit counts as a cache hit in ai_usage().
bool AICacheProbe(const AIRequest &request, const string &query_text, AIResult &out) {
	const auto &config = AIConfig::Get();
	if (!config.local_cache) {
		return false;
	}
	idx_t query_index;
	{
		std::lock_guard<std::mutex> lock(g_usage_mutex);
		query_index = GetOrCreateQueryIndex(query_text);
	}
	const string key = CacheScopePrefix(config, query_index) + RequestSignature(config, request);
	{
		std::lock_guard<std::mutex> lock(g_cache_mutex);
		auto it = g_cache.find(key);
		if (it == g_cache.end()) {
			return false;
		}
		out = it->second;
	}
	{
		std::lock_guard<std::mutex> lock(g_usage_mutex);
		g_queries[query_index].cache_hits++;
	}
	return true;
}

static void RunConcurrent(idx_t n, idx_t max_workers, const std::function<void(idx_t)> &fn) {
	if (n == 0) {
		return;
	}
	const idx_t worker_count = std::min<idx_t>(max_workers, n);
	if (worker_count <= 1) {
		for (idx_t i = 0; i < n; i++) {
			fn(i);
		}
		return;
	}
	std::atomic<idx_t> next {0};
	vector<std::thread> workers;
	workers.reserve(worker_count);
	for (idx_t w = 0; w < worker_count; w++) {
		workers.emplace_back([&]() {
			for (;;) {
				const idx_t i = next.fetch_add(1);
				if (i >= n) {
					break;
				}
				fn(i);
			}
		});
	}
	for (auto &worker : workers) {
		worker.join();
	}
}

//===--------------------------------------------------------------------===//
// Turbo mode: AIMD adaptive concurrency
//===--------------------------------------------------------------------===//
//! Concurrency limiter with additive-increase / multiplicative-decrease control.
//! `limit` is the max number of in-flight requests; it grows by 1 after a streak of
//! successes and halves on a throttle (429/503/529), staying within [min_limit, max_limit].
class AdaptiveLimiter {
public:
	AdaptiveLimiter(idx_t start, idx_t min_l, idx_t max_l)
	    : min_limit(min_l < 1 ? 1 : min_l), max_limit(max_l < min_limit ? min_limit : max_l) {
		limit = start < min_limit ? min_limit : (start > max_limit ? max_limit : start);
	}

	//! Block until an in-flight slot is free, then occupy it.
	void Acquire() {
		std::unique_lock<std::mutex> lock(mu);
		cv.wait(lock, [&]() { return in_flight < limit; });
		in_flight++;
	}

	void Release() {
		std::lock_guard<std::mutex> lock(mu);
		if (in_flight > 0) {
			in_flight--;
		}
		cv.notify_one();
	}

	//! Additive increase: after `limit` successes, widen the window by one (gentle, TCP-like).
	void OnSuccess() {
		std::lock_guard<std::mutex> lock(mu);
		if (limit >= max_limit) {
			return;
		}
		if (++success_streak >= limit) {
			success_streak = 0;
			limit++;
			cv.notify_all(); // a new slot just opened
		}
	}

	//! Multiplicative decrease: halve the window. A short cooldown collapses a burst of
	//! simultaneous throttles (many in-flight requests all getting 429) into one halving.
	void OnThrottle() {
		std::lock_guard<std::mutex> lock(mu);
		success_streak = 0;
		const auto now = std::chrono::steady_clock::now();
		if (now - last_decrease < std::chrono::milliseconds(250)) {
			return;
		}
		last_decrease = now;
		const idx_t halved = limit / 2;
		limit = halved < min_limit ? min_limit : halved;
	}

private:
	std::mutex mu;
	std::condition_variable cv;
	idx_t limit;
	idx_t in_flight = 0;
	idx_t success_streak = 0;
	const idx_t min_limit;
	const idx_t max_limit;
	std::chrono::steady_clock::time_point last_decrease {}; // epoch => first throttle always decreases
};

//! Backoff before retrying a throttled request. Uses full jitter (a uniform point in the growing
//! exponential window) so concurrent retries decorrelate instead of colliding in lockstep waves;
//! any server Retry-After hint is honored as a floor on top of the jitter.
static void SleepBackoff(idx_t attempt, int32_t retry_after_ms, idx_t seed) {
	// Exponential window: 100ms, 200, 400, ... capped at 8s.
	const idx_t shift = attempt > 7 ? 6 : (attempt == 0 ? 0 : attempt - 1);
	const int64_t window = std::min<int64_t>(static_cast<int64_t>(100) << shift, 8000);
	// Hash (seed, attempt) -> uniform in [0, window): decorrelates rows/attempts without shared RNG state.
	uint64_t h = static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(attempt) * 0xD1B54A32D192ED03ULL;
	h ^= h >> 33;
	h *= 0xFF51AFD7ED558CCDULL;
	h ^= h >> 33;
	const int64_t jitter = static_cast<int64_t>(h % static_cast<uint64_t>(window));
	const int64_t floor_ms = retry_after_ms >= 0 ? retry_after_ms : 0;
	std::this_thread::sleep_for(std::chrono::milliseconds(floor_ms + jitter));
}

//! Perform one request, retrying with backoff on throttle (429/503/529) AND transient failure
//! (network error / 5xx / unparseable body). Permanent 4xx errors are returned without retry.
//! When `limiter` is set (turbo), gate each attempt through it and feed it AIMD signals; otherwise
//! concurrency is fixed by the caller's thread pool and only the retry loop applies.
//! A request given up on: count it (ai_usage().failed_calls) and say so ONCE per endpoint + failure
//! kind on stderr. Before this, an unreachable endpoint produced NULLs and a zero-call ai_usage()
//! with no message at all -- indistinguishable from "nothing to do".
static void NoteCallFailure(const AIConfig &config, const AIRequest &request, idx_t query_index, const string &error) {
	{
		std::lock_guard<std::mutex> lock(g_usage_mutex);
		g_queries[query_index].failed_calls++;
	}
	static std::mutex reported_mutex;
	static std::set<string> reported;
	const string endpoint = AIUsesTypeSafe(request) ? config.typesafe_url : config.base_url;
	std::lock_guard<std::mutex> lock(reported_mutex);
	if (reported.insert(endpoint + "\x1f" + error.substr(0, 32)).second) {
		fprintf(stderr,
		        "[aisql] LLM request to %s failed: %s\n[aisql] The AI function returns NULL for the affected rows; "
		        "ai_usage().failed_calls counts them. Is the endpoint running (serve/start_stack.sh)? "
		        "SET ai_endpoint / AI_PROXY_URL selects it.\n",
		        endpoint.c_str(), error.c_str());
	}
}

static AIResult DoRequestWithRetry(const AIConfig &config, const AIRequest &request, idx_t query_index, idx_t seed,
                                   AdaptiveLimiter *limiter) {
	idx_t attempt = 0;
	for (;;) {
		if (limiter) {
			limiter->Acquire();
		}
		AICallOutcome outcome;
		int32_t retry_after_ms;
		string error;
		AIResult result = DoSingleRequest(config, request, query_index, outcome, retry_after_ms, error);
		if (limiter) {
			limiter->Release();
		}
		if (outcome == AICallOutcome::THROTTLED || outcome == AICallOutcome::TRANSIENT) {
			// A throttle is a rate signal (shrink the AIMD window); a transient error is not.
			if (limiter && outcome == AICallOutcome::THROTTLED) {
				limiter->OnThrottle();
			}
			if (attempt >= config.max_retries) {
				NoteCallFailure(config, request, query_index,
				                error + " (after " + std::to_string(attempt + 1) + " attempt(s); ai_max_retries=" +
				                    std::to_string(config.max_retries) + ")");
				return result; // exhausted retries: keep the failure result
			}
			attempt++;
			SleepBackoff(attempt, retry_after_ms, seed);
			continue;
		}
		if (limiter && outcome == AICallOutcome::OK) {
			limiter->OnSuccess();
		}
		if (outcome == AICallOutcome::ERROR) {
			NoteCallFailure(config, request, query_index, error); // permanent (4xx, no connection possible)
		}
		return result; // OK or permanent error (neutral to the AIMD window)
	}
}

//===--------------------------------------------------------------------===//
// Hedged requests: tail-latency insurance. A call still unanswered past the hedge deadline --
// 1.2x the observed p99 of winning-call latencies, floored at 2s, armed only after 50
// samples -- fires ONE duplicate attempt; whichever lands first wins and the loser's response
// is discarded (its usage still accounts: both calls really happened). The floor and margin
// keep the mock and uniform-latency benches hedge-free, so call-envelope tests are unaffected;
// on real APIs this converts p99+ stragglers (the 920s q12 outlier class) into ~p50 waits at
// ~1% extra calls.
//===--------------------------------------------------------------------===//
static std::mutex g_lat_mutex;
static vector<int64_t> g_lat_ring;
static idx_t g_lat_pos = 0;

static void RecordCallLatency(int64_t ms) {
	std::lock_guard<std::mutex> lock(g_lat_mutex);
	if (g_lat_ring.size() < 512) {
		g_lat_ring.push_back(ms);
	} else {
		g_lat_ring[g_lat_pos] = ms;
		g_lat_pos = (g_lat_pos + 1) % g_lat_ring.size();
	}
}

static int64_t HedgeDeadlineMs() {
	vector<int64_t> sample;
	{
		std::lock_guard<std::mutex> lock(g_lat_mutex);
		if (g_lat_ring.size() < 50) {
			return -1;
		}
		sample = g_lat_ring;
	}
	const idx_t p99_pos = (sample.size() * 99) / 100;
	std::nth_element(sample.begin(), sample.begin() + NumericCast<int64_t>(p99_pos), sample.end());
	const int64_t p99 = sample[p99_pos];
	// FLOOR 60s, not 2s. A hedge is a DUPLICATE call: it buys tail latency with money and with
	// provider load, so it must only fire when a request is genuinely stuck, not merely slower
	// than its peers. At 2s the floor fired constantly on a long query -- agent_bench Q19 spent
	// 103-264 calls on hedges across runs, and every call SWAN made beyond PLOP's 9,537 was a
	// hedge duplicate rather than any difference in what the optimizer evaluated. The in-flight
	// dedup doubled it (103 -> 226) simply by making waiters block, which raised observed latency
	// and pushed more requests past the deadline.
	return MaxValue<int64_t>(60000, (p99 * 12) / 10);
}

//===--------------------------------------------------------------------===//
// Prefix lifecycle: the client-side cache-write policy for explicit provider prompt caching.
// Operators only DECLARE structure (prompt_prefix = where sameness ends, expected_reuse = known
// fan-out); this state machine decides whether/when a breakpoint is emitted:
//   COLD       first sight. Hint >= 2 -> this call is the WRITE (prime). No hint -> send plain,
//              remember (adaptive: the bet is only placed once reuse is a proven fact).
//   SEEN_PLAIN second arrival of an unhinted prefix -> this call is the WRITE.
//   PRIMING    a write is in flight. Same-prefix calls PARK (holding no concurrency permit --
//              the ChatGatePermit is taken later, inside DoSingleRequest) until it lands:
//              concurrent same-prefix requests race past an unfinished write and all miss
//              (measured 50% vs 99% cached).
//   WARM       reads (0.1x) while fresh; the provider TTL slides on use, so last use past
//              ~25 min makes the next call re-prime.
//   NOCACHE    the write failed: fail open, everything goes plain (per process).
//===--------------------------------------------------------------------===//
enum class PrefixPhase : uint8_t { SEEN_PLAIN, PRIMING, WARM, NOCACHE };
struct PrefixState {
	PrefixPhase phase;
	std::chrono::steady_clock::time_point last_use;
};
static std::mutex g_prefix_mutex;
static std::condition_variable g_prefix_cv;
static std::unordered_map<string, PrefixState> g_prefix_states;
static constexpr int64_t PREFIX_TTL_SECONDS = 25 * 60; // provider keeps ~30 min after last use

//! Provider cache identity of the reusable span. model/system/schema conservatively included:
//! colliding spans under different envelopes would at worst prime twice, never corrupt.
static string PrefixKey(const AIConfig &config, const AIRequest &request) {
	return config.model + "\x1f" + request.system_prompt + "\x1f" + request.json_schema + "\x1f" +
	       request.prompt_prefix;
}

//! Decide this request's emit_breakpoint (parking through PRIMING). Returns true when the caller
//! became the prime and owes FinishPrefixPrime() after its request completes.
static bool ApplyPrefixLifecycle(const AIConfig &config, AIRequest &request) {
	// plain variant: never split the prompt into cache blocks -- a split body sends the user
	// content as an ARRAY of parts, and a bare-prompt comparison needs it to stay a single string.
	if (!config.prefix_cache || config.prompt_variant_plain || request.prompt_prefix.empty()) {
		return false;
	}
	// Size gate: system + prefix must clear the provider's 1,024-visible-token minimum. Biased
	// toward attempting (a too-short prefix is ignored at zero cost; a skipped long one forfeits
	// the read discount); the text ratio self-calibrates from observed usage.
	const double prefix_tokens = AITextHasImage(request.prompt_prefix)
	                                 ? AIEstimatePromptCost(request.prompt_prefix)
	                                 : static_cast<double>(request.prompt_prefix.size()) / AICalibratedBytesPerToken();
	if (prefix_tokens < 900.0) {
		return false;
	}
	const string key = PrefixKey(config, request);
	const auto now = std::chrono::steady_clock::now();
	std::unique_lock<std::mutex> lock(g_prefix_mutex);
	while (true) {
		auto it = g_prefix_states.find(key);
		if (it == g_prefix_states.end()) {
			if (request.expected_reuse >= 2) {
				g_prefix_states[key] = {PrefixPhase::PRIMING, now};
				request.emit_breakpoint = true;
				return true; // hinted: first call is the write
			}
			g_prefix_states[key] = {PrefixPhase::SEEN_PLAIN, now};
			return false; // adaptive: first arrival goes plain
		}
		switch (it->second.phase) {
		case PrefixPhase::SEEN_PLAIN:
			it->second = {PrefixPhase::PRIMING, now};
			request.emit_breakpoint = true;
			return true; // adaptive: proven reuse, second arrival is the write
		case PrefixPhase::PRIMING:
			g_prefix_cv.wait(lock); // park (no permit held) until the write lands or fails
			continue;
		case PrefixPhase::WARM:
			if (std::chrono::duration_cast<std::chrono::seconds>(now - it->second.last_use).count() >
			    PREFIX_TTL_SECONDS) {
				it->second = {PrefixPhase::PRIMING, now};
				request.emit_breakpoint = true;
				return true; // stale: this call re-primes
			}
			it->second.last_use = now;
			request.emit_breakpoint = true;
			return false; // read at 0.1x
		case PrefixPhase::NOCACHE:
			return false; // fail-open: plain
		}
	}
}

static void FinishPrefixPrime(const AIConfig &config, const AIRequest &request, bool success) {
	const string key = PrefixKey(config, request);
	{
		std::lock_guard<std::mutex> lock(g_prefix_mutex);
		g_prefix_states[key] = {success ? PrefixPhase::WARM : PrefixPhase::NOCACHE,
		                        std::chrono::steady_clock::now()};
	}
	g_prefix_cv.notify_all();
}

static AIResult HedgedRequest(const AIConfig &config, const AIRequest &request, idx_t query_index, idx_t seed,
                              AdaptiveLimiter *limiter) {
	// Prefix lifecycle first: may PARK here (before any latency clock or permit) until an
	// in-flight cache write lands, and stamps emit_breakpoint on a local copy for this dispatch.
	AIRequest dispatch = request;
	// System One has no provider prefix cache: never park behind or prime one for it.
	const bool is_prime = !AIUsesTypeSafe(request) && ApplyPrefixLifecycle(config, dispatch);
	const auto t0 = std::chrono::steady_clock::now();
	const int64_t deadline_ms = config.hedge ? HedgeDeadlineMs() : -1;
	AIResult winner;
	if (deadline_ms < 0) {
		winner = DoRequestWithRetry(config, dispatch, query_index, seed, limiter);
	} else {
		struct Shared {
			std::mutex m;
			std::condition_variable cv;
			bool done = false;
			AIResult result;
		};
		auto shared = std::make_shared<Shared>();
		const AIConfig *config_ptr = &config; // process-global singleton: safe beyond this frame
		auto launch = [config_ptr, dispatch, query_index, limiter, shared](idx_t attempt_seed) {
			std::thread([config_ptr, dispatch, query_index, limiter, shared, attempt_seed]() {
				AIResult r = DoRequestWithRetry(*config_ptr, dispatch, query_index, attempt_seed, limiter);
				std::lock_guard<std::mutex> lock(shared->m);
				if (!shared->done) {
					shared->done = true;
					shared->result = std::move(r);
					shared->cv.notify_all();
				}
			}).detach();
		};
		launch(seed);
		std::unique_lock<std::mutex> lock(shared->m);
		if (!shared->cv.wait_for(lock, std::chrono::milliseconds(deadline_ms), [&]() { return shared->done; })) {
			lock.unlock();
			{
				std::lock_guard<std::mutex> ulock(g_usage_mutex);
				g_queries[query_index].hedged_calls++;
			}
			launch(seed ^ 0x9E3779B9U);
			lock.lock();
		}
		shared->cv.wait(lock, [&]() { return shared->done; });
		winner = shared->result;
	}
	if (is_prime) {
		FinishPrefixPrime(config, dispatch, winner.success); // wake parked same-prefix readers
	}
	RecordCallLatency(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
	return winner;
}

vector<AIResult> AIBatchComplete(const vector<AIRequest> &requests, const string &query_text, bool force_fixed) {
	const auto &config = AIConfig::Get();
	const idx_t n = requests.size();
	vector<AIResult> results(n);
	if (n == 0) {
		return results;
	}

	idx_t query_index;
	{
		std::lock_guard<std::mutex> lock(g_usage_mutex);
		query_index = GetOrCreateQueryIndex(query_text);
	}

	// Cache keys are scoped to this query (query_index prefix), so the response cache is
	// query-local: identical prompts are de-duplicated within a query (across its chunks),
	// but one query never serves cached results to another.
	vector<string> keys(n);
	const string query_scope = CacheScopePrefix(config, query_index);
	for (idx_t i = 0; i < n; i++) {
		keys[i] = query_scope + RequestSignature(config, requests[i]);
	}

	// Serve cache hits, join whatever is already in flight, and collect the rest as unique misses.
	vector<idx_t> miss_rows;        // representative request index per unique miss THIS batch owns
	vector<idx_t> row_to_unique(n); // maps each row to its unique-miss slot
	vector<bool> served(n, false);
	vector<bool> waiting(n, false);
	vector<shared_ptr<AIInFlight>> row_wait(n);
	AIInFlightPublisher publisher; // retires this batch's in-flight entries on every exit path
	{
		std::lock_guard<std::mutex> lock(g_cache_mutex);
		std::unordered_map<string, idx_t> local_unique;
		for (idx_t i = 0; i < n; i++) {
			auto cached = config.local_cache ? g_cache.find(keys[i]) : g_cache.end();
			if (cached != g_cache.end()) {
				results[i] = cached->second;
				served[i] = true;
				continue;
			}
			auto seen = local_unique.find(keys[i]);
			if (seen != local_unique.end()) {
				row_to_unique[i] = seen->second;
				continue;
			}
			if (config.local_cache) {
				// Another batch already sent this exact prompt: wait for its answer instead of
				// asking the same question twice. Registered under the same key, so the query
				// scope applies here too.
				auto flying = g_inflight.find(keys[i]);
				if (flying != g_inflight.end()) {
					row_wait[i] = flying->second;
					waiting[i] = true;
					continue;
				}
			}
			const idx_t slot = miss_rows.size();
			local_unique.emplace(keys[i], slot);
			miss_rows.push_back(i);
			row_to_unique[i] = slot;
			if (config.local_cache) {
				auto entry = make_shared_ptr<AIInFlight>();
				g_inflight.emplace(keys[i], entry);
				publisher.Add(std::move(entry), keys[i]);
			}
		}
	}

	// Every input row that did not spawn a unique network request is a cache/de-dup hit.
	{
		std::lock_guard<std::mutex> lock(g_usage_mutex);
		g_queries[query_index].cache_hits += (n - miss_rows.size());
	}

	// Execute the unique misses concurrently. Concurrency never exceeds the number of
	// requests in this chunk, so it is bounded by the data chunk size either way. Both modes
	// retry throttled requests; turbo additionally adapts the concurrency window (AIMD).
	vector<AIResult> miss_results(miss_rows.size());
	if (AITurboEnabled() && !force_fixed) {
		const idx_t turbo_max = std::min<idx_t>(config.turbo_max, miss_rows.size());
		AdaptiveLimiter limiter(config.turbo_start, config.turbo_min, turbo_max);
		RunConcurrent(miss_rows.size(), turbo_max, [&](idx_t u) {
			miss_results[u] = HedgedRequest(config, requests[miss_rows[u]], query_index, u, &limiter);
		});
	} else {
		RunConcurrent(miss_rows.size(), config.max_concurrency, [&](idx_t u) {
			miss_results[u] = HedgedRequest(config, requests[miss_rows[u]], query_index, u, nullptr);
		});
	}

	// Cache successes only (so transient failures can be retried later). Done BEFORE the
	// in-flight entries retire, so a batch arriving in the gap finds the cached answer rather
	// than re-sending: every instant is covered by one or the other.
	{
		std::lock_guard<std::mutex> lock(g_cache_mutex);
		for (idx_t u = 0; u < miss_rows.size(); u++) {
			if (config.local_cache && miss_results[u].success) {
				g_cache[keys[miss_rows[u]]] = miss_results[u];
			}
		}
	}
	publisher.Publish(miss_results);

	for (idx_t i = 0; i < n; i++) {
		if (!served[i] && !waiting[i]) {
			results[i] = miss_results[row_to_unique[i]];
		}
	}
	// Only now block on prompts another batch owns. Waiting AFTER publishing our own is what
	// keeps this deadlock-free: two batches holding each other's keys both publish before either
	// waits, so neither can be blocked by the other.
	for (idx_t i = 0; i < n; i++) {
		if (!waiting[i]) {
			continue;
		}
		auto &entry = *row_wait[i];
		std::unique_lock<std::mutex> lock(entry.m);
		entry.cv.wait(lock, [&entry]() { return entry.done; });
		results[i] = entry.result;
	}
	return results;
}

//===--------------------------------------------------------------------===//
// Embeddings: ai_embed (fixed path, no history; one batched request per chunk)
//===--------------------------------------------------------------------===//
static std::mutex g_embed_cache_mutex;
static std::unordered_map<string, vector<float>> g_embed_cache; // query-scoped key -> vector

//! An embed input that IS a single ai_image sentinel: sent as an image item ({"image": ref}).
static bool IsImageEmbedInput(const string &in) {
	return in.size() > 2 && in.front() == AI_IMAGE_OPEN && in.back() == AI_IMAGE_CLOSE &&
	       in.find(AI_IMAGE_OPEN, 1) == string::npos;
}

//! Build an OpenAI-style embeddings body. temperature=-1 marks this as an ai_embed call and keeps
//! its cache key distinct from any chat request for the same text.
static string BuildEmbedBody(const string &model, const vector<string> &inputs) {
	string body = "{\"model\":\"" + AIJsonEscape(model) + "\",\"temperature\":-1,\"input\":[";
	for (idx_t i = 0; i < inputs.size(); i++) {
		if (i > 0) {
			body += ",";
		}
		// An input that IS a single ai_image sentinel becomes an image item ({"image": ref}) for
		// the dual-encoder server; everything else embeds as text.
		const auto &in = inputs[i];
		if (IsImageEmbedInput(in)) {
			// A local path is relative to THIS process's working directory (where the chat path reads the
			// file); the server runs elsewhere, so send it absolute.
			string ref = in.substr(1, in.size() - 2);
			const bool remote = ref.compare(0, 5, "data:") == 0 || ref.compare(0, 7, "http://") == 0 ||
			                    ref.compare(0, 8, "https://") == 0;
#ifdef _WIN32
			const bool absolute = ref.size() > 1 && (ref[1] == ':' || ref[0] == '\\' || ref[0] == '/');
#else
			const bool absolute = !ref.empty() && ref[0] == '/';
#endif
			if (!remote && !ref.empty() && !absolute) {
				ref = FileSystem::GetWorkingDirectory() + "/" + ref;
			}
			body += "{\"image\":\"" + AIJsonEscape(ref) + "\"}";
		} else if (!in.empty() && in.front() == AI_IMAGE_TEXT_MARK) {
			body += "{\"image_text\":\"" + AIJsonEscape(in.substr(1)) + "\"}";
		} else {
			body += "\"" + AIJsonEscape(in) + "\"";
		}
	}
	body += "]}";
	return body;
}

// Latched the first time the embeddings server answers an image item without an embedding (text-only
// model): from then on image leaves keep their neutral prior and no request carries image items.
static std::atomic<bool> g_embed_images_unsupported {false};

bool AIEmbedImagesSupported() {
	return AIConfig::Get().embed_images && !g_embed_images_unsupported.load();
}

//! POST one embeddings request for `inputs`; fill `out[i]` with the i-th embedding. Returns false on
//! any transport/parse failure. An item the server answers without an embedding (an image the model
//! cannot embed, an unreadable file) leaves out[i] empty without failing the batch -- the text items
//! beside it still land. Accounts one llm_call + usage tokens against the query.
static bool DoEmbedBatchRequest(const AIConfig &config, const vector<string> &inputs, idx_t query_index,
                                vector<vector<float>> &out) {
	out.assign(inputs.size(), {});
	ConnLease lease(config.embed_url, config);

	duckdb_httplib::Headers headers;
	if (!config.api_key.empty()) {
		headers.emplace("Authorization", "Bearer " + config.api_key);
	}
	const auto start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                          std::chrono::system_clock::now().time_since_epoch())
	                          .count();
	headers.emplace("X-Request-Start-Ms", std::to_string(start_ms));
	const string body = BuildEmbedBody(config.embed_model, inputs);
	auto response = lease.client->Post("/v1/embeddings", headers, body, "application/json");
	if (!response && lease.reused) {
		lease.Refresh(); // stale keep-alive socket: one silent retry on a fresh connection
		response = lease.client->Post("/v1/embeddings", headers, body, "application/json");
	}
	if (!response) {
		return false;
	}
	lease.healthy = true;
	if (response->status < 200 || response->status >= 300) {
		return false;
	}

	yyjson_doc *doc = yyjson_read(response->body.c_str(), response->body.size(), 0);
	if (!doc) {
		return false;
	}
	bool ok = false;
	yyjson_val *root = yyjson_doc_get_root(doc);
	yyjson_val *data = root && yyjson_is_obj(root) ? yyjson_obj_get(root, "data") : nullptr;
	if (data && yyjson_is_arr(data)) {
		ok = true;
		size_t idx, max;
		yyjson_val *item;
		yyjson_arr_foreach(data, idx, max, item) {
			// Respect an explicit "index" field if present, else fall back to array order.
			idx_t pos = idx;
			yyjson_val *jindex = yyjson_obj_get(item, "index");
			if (jindex && yyjson_is_int(jindex)) {
				pos = static_cast<idx_t>(yyjson_get_int(jindex));
			}
			if (pos >= out.size()) {
				ok = false;
				break;
			}
			yyjson_val *emb = yyjson_obj_get(item, "embedding");
			if (!emb || !yyjson_is_arr(emb)) {
				if (IsImageEmbedInput(inputs[pos])) {
					g_embed_images_unsupported.store(true);
				}
				continue; // error entry: this item stays empty, the rest of the batch is fine
			}
			vector<float> vec;
			vec.reserve(yyjson_arr_size(emb));
			size_t ei, emax;
			yyjson_val *num;
			yyjson_arr_foreach(emb, ei, emax, num) {
				vec.push_back(yyjson_is_num(num) ? static_cast<float>(yyjson_get_num(num)) : 0.0f);
			}
			out[pos] = std::move(vec);
		}
	}
	// Usage: one request; embeddings are local/free, so cost stays 0.
	uint64_t total_tokens = 0;
	yyjson_val *usage = root ? yyjson_obj_get(root, "usage") : nullptr;
	if (usage && yyjson_is_obj(usage)) {
		yyjson_val *tt = yyjson_obj_get(usage, "total_tokens");
		if (tt && yyjson_is_num(tt)) {
			total_tokens = static_cast<uint64_t>(yyjson_get_num(tt));
		}
	}
	yyjson_doc_free(doc);
	if (ok) {
		std::lock_guard<std::mutex> lock(g_usage_mutex);
		auto &record = g_queries[query_index];
		record.llm_calls += 1;
		record.embed_calls += 1; // so chat calls == llm_calls - embed_calls (process-isolated count)
		record.input_tokens += total_tokens;
		record.embed_tokens += total_tokens;
		record.total_tokens += total_tokens;
	}
	return ok;
}

vector<AIEmbedResult> AIEmbedBatch(const vector<string> &texts, const string &query_text) {
	const auto &config = AIConfig::Get();
	const idx_t n = texts.size();
	vector<AIEmbedResult> results(n);
	if (n == 0) {
		return results;
	}

	idx_t query_index;
	{
		std::lock_guard<std::mutex> lock(g_usage_mutex);
		query_index = GetOrCreateQueryIndex(query_text);
	}

	// Query-scoped, embed-namespaced cache keys (no chat/embed collision; no cross-query sharing).
	vector<string> keys(n);
	const string scope = CacheScopePrefix(config, query_index) + "emb" + "\x1f" + config.embed_model + "\x1f";
	for (idx_t i = 0; i < n; i++) {
		keys[i] = scope + texts[i];
	}

	// De-dup within the chunk + serve query cache hits.
	vector<idx_t> miss_rows;
	vector<idx_t> row_to_unique(n);
	vector<bool> served(n, false);
	{
		std::lock_guard<std::mutex> lock(g_embed_cache_mutex);
		std::unordered_map<string, idx_t> local_unique;
		for (idx_t i = 0; i < n; i++) {
			if (texts[i].empty()) {
				served[i] = true; // leave results[i].success = false -> SQL NULL
				continue;
			}
			auto cached = config.local_cache ? g_embed_cache.find(keys[i]) : g_embed_cache.end();
			if (cached != g_embed_cache.end()) {
				results[i].embedding = cached->second;
				results[i].success = true;
				served[i] = true;
				continue;
			}
			auto seen = local_unique.find(keys[i]);
			if (seen != local_unique.end()) {
				row_to_unique[i] = seen->second;
				continue;
			}
			const idx_t slot = miss_rows.size();
			local_unique.emplace(keys[i], slot);
			miss_rows.push_back(i);
			row_to_unique[i] = slot;
		}
	}
	{
		std::lock_guard<std::mutex> lock(g_usage_mutex);
		g_queries[query_index].cache_hits += (n - miss_rows.size());
	}

	// Batched requests for the unique misses, split into sub-batches (<= embed_batch_images image items,
	// <= 512 items) and issued embed_concurrency at a time. Splitting keeps every request well inside the
	// client timeout (a CLIP image encode is ~0.2 s on a CPU: one 500-image request -- an ECOMM table --
	// took 80 s, timed out, and every feature of the chunk silently came back empty); concurrency is
	// what the server's per-request threads and torch's intra-op pool reward (~2x throughput at 4-8
	// in flight). A sub-batch that fails blanks only its own items.
	constexpr idx_t kMaxItemsPerRequest = 512;
	const idx_t max_images = MaxValue<idx_t>(config.embed_batch_images, 1);
	vector<std::pair<idx_t, idx_t>> ranges; // [start, end) into miss_rows
	for (idx_t start = 0; start < miss_rows.size();) {
		idx_t images = 0;
		idx_t end = start;
		for (; end < miss_rows.size() && end - start < kMaxItemsPerRequest; end++) {
			const bool image = IsImageEmbedInput(texts[miss_rows[end]]);
			if (image && images == max_images) {
				break;
			}
			images += image ? 1 : 0;
		}
		ranges.emplace_back(start, end);
		start = end;
	}
	vector<vector<float>> miss_emb(miss_rows.size());
	std::atomic<idx_t> next_range {0};
	auto worker = [&]() {
		for (;;) {
			const idx_t r = next_range.fetch_add(1);
			if (r >= ranges.size()) {
				return;
			}
			const idx_t start = ranges[r].first, end = ranges[r].second;
			vector<string> inputs;
			inputs.reserve(end - start);
			for (idx_t u = start; u < end; u++) {
				inputs.push_back(texts[miss_rows[u]]);
			}
			vector<vector<float>> part;
			bool ok = false;
			try {
				ok = DoEmbedBatchRequest(config, inputs, query_index, part);
			} catch (...) {
				ok = false; // a throwing worker thread would terminate the process; a failed sub-batch is NULL
			}
			if (ok) {
				std::lock_guard<std::mutex> lock(g_embed_cache_mutex);
				for (idx_t u = 0; u < inputs.size(); u++) {
					if (config.local_cache && !part[u].empty()) {
						g_embed_cache[keys[miss_rows[start + u]]] = part[u];
					}
					miss_emb[start + u] = std::move(part[u]);
				}
			}
		}
	};
	const idx_t nthreads = MinValue<idx_t>(MaxValue<idx_t>(config.embed_concurrency, 1), ranges.size());
	if (nthreads <= 1) {
		worker();
	} else {
		vector<std::thread> pool;
		for (idx_t t = 0; t < nthreads; t++) {
			pool.emplace_back(worker);
		}
		for (auto &t : pool) {
			t.join();
		}
	}

	for (idx_t i = 0; i < n; i++) {
		if (served[i]) {
			continue;
		}
		const idx_t u = row_to_unique[i];
		if (u < miss_emb.size() && !miss_emb[u].empty()) {
			results[i].embedding = miss_emb[u];
			results[i].success = true;
		}
	}
	return results;
}

//===--------------------------------------------------------------------===//
// Usage reporting
//===--------------------------------------------------------------------===//
uint64_t AILocalCacheClear() {
	uint64_t cleared = 0;
	{
		std::lock_guard<std::mutex> lock(g_cache_mutex);
		cleared += g_cache.size();
		g_cache.clear();
	}
	{
		std::lock_guard<std::mutex> lock(g_embed_cache_mutex);
		cleared += g_embed_cache.size();
		g_embed_cache.clear();
	}
	return cleared;
}

vector<AIQueryUsage> AIGetUsage() {
	std::lock_guard<std::mutex> lock(g_usage_mutex);
	return g_queries;
}

uint64_t AIResetUsage() {
	std::lock_guard<std::mutex> lock(g_usage_mutex);
	const uint64_t cleared = g_queries.size();
	g_queries.clear();
	g_query_index.clear();
	g_next_query_id = 1;
	return cleared;
}

//===--------------------------------------------------------------------===//
// Training data: (prompt, embedding, actual ai_filter result), deduped by prompt
//===--------------------------------------------------------------------===//
static std::mutex g_training_mutex;
static std::unordered_map<string, AITrainingExample> g_training;

void AIRecordTrainingExample(const string &prompt, const vector<float> &embedding, bool label) {
	std::lock_guard<std::mutex> lock(g_training_mutex);
	if (g_training.find(prompt) != g_training.end()) {
		return; // dedup by prompt: keep the first observed label
	}
	g_training.emplace(prompt, AITrainingExample {prompt, embedding, label});
}

vector<AITrainingExample> AIGetTrainingData() {
	std::lock_guard<std::mutex> lock(g_training_mutex);
	vector<AITrainingExample> out;
	out.reserve(g_training.size());
	for (auto &kv : g_training) {
		out.push_back(kv.second);
	}
	return out;
}

uint64_t AIResetTrainingData() {
	std::lock_guard<std::mutex> lock(g_training_mutex);
	const uint64_t cleared = g_training.size();
	g_training.clear();
	return cleared;
}

//===--------------------------------------------------------------------===//
// History mode: per-value conversation memory (session-lived; reset explicitly)
//===--------------------------------------------------------------------===//
struct AIHistoryTurn {
	string input;
	string output;
};
static std::mutex g_history_mutex;
static std::unordered_map<string, vector<AIHistoryTurn>> g_history;
static std::atomic<bool> g_history_enabled {false};

void AIHistorySetEnabled(bool enabled) {
	g_history_enabled.store(enabled);
}

bool AIHistoryEnabled() {
	return g_history_enabled.load();
}

void AIHistoryRecord(const vector<string> &keys, const string &input, const string &output) {
	if (keys.empty()) {
		return;
	}
	std::lock_guard<std::mutex> lock(g_history_mutex);
	for (const auto &key : keys) {
		g_history[key].push_back(AIHistoryTurn {input, output});
	}
}

vector<std::pair<string, string>> AIHistoryGather(const vector<string> &keys) {
	vector<std::pair<string, string>> turns;
	if (keys.empty()) {
		return turns;
	}
	std::lock_guard<std::mutex> lock(g_history_mutex);
	std::set<std::pair<string, string>> seen;
	for (const auto &key : keys) {
		auto it = g_history.find(key);
		if (it == g_history.end()) {
			continue;
		}
		for (const auto &turn : it->second) {
			auto pair = std::make_pair(turn.input, turn.output);
			if (seen.insert(pair).second) {
				turns.push_back(std::move(pair));
			}
		}
	}
	return turns;
}

uint64_t AIHistoryReset() {
	std::lock_guard<std::mutex> lock(g_history_mutex);
	const uint64_t cleared = g_history.size();
	g_history.clear();
	return cleared;
}

} // namespace duckdb
