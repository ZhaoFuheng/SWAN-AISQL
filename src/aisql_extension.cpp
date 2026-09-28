//===----------------------------------------------------------------------===//
// aisql: semantic SQL for DuckDB (SWAN AI-SQL v2)
//
// Entry point: registers settings, AI scalar/table functions, the in-process
// mock backend, and the optimizer pipeline. See DESIGN in the repo plan.
//===----------------------------------------------------------------------===//
#include "aisql_extension.hpp"

#include "ai_client.hpp"
#include "ai_functions.hpp"
#include "ai_mock_server.hpp"
#include "optimizer/aisql_optimizer.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <csignal>
#include <cstdlib>

namespace duckdb {

//! Env fallback for settings without an AIConfig field: default seeded from the legacy env var.
static Value EnvOr(const char *env, const Value &def) {
	const char *v = std::getenv(env);
	if (!v || !v[0]) {
		return def;
	}
	return Value(v).DefaultCastAs(def.type());
}

//! Legacy on-unless-off env flag (fork semantics): unset or any value -> def; "off"/"0" -> false.
static Value EnvOnUnlessOff(const char *env, bool def) {
	const char *v = std::getenv(env);
	if (!v || !v[0]) {
		return Value::BOOLEAN(def);
	}
	const string s(v);
	return Value::BOOLEAN(!(s == "off" || s == "0"));
}

//! SET-callback bridge into the client's process-global AIConfig (env vars seed its initial
//! values inside AIConfig::Get(); a SET afterwards wins).
static void RegisterAISettings(DatabaseInstance &db) {
	auto &config = DBConfig::GetConfig(db);
	const auto &ai = AIConfig::Get(); // force env seeding before defaults are read
	// Connection surface (defaults mirror the seeded AIConfig; SET updates it live)
	config.AddExtensionOption(
	    "ai_endpoint", "OpenAI-compatible LLM endpoint", LogicalType::VARCHAR, Value(ai.base_url),
	    [](ClientContext &, SetScope, Value &v) { AIConfig::Mutable().base_url = StringValue::Get(v); });
	config.AddExtensionOption(
	    "ai_model", "LLM model id", LogicalType::VARCHAR, Value(ai.model),
	    [](ClientContext &, SetScope, Value &v) { AIConfig::Mutable().model = StringValue::Get(v); });
	config.AddExtensionOption(
	    "ai_api_key", "LLM API key", LogicalType::VARCHAR, Value(ai.api_key),
	    [](ClientContext &, SetScope, Value &v) { AIConfig::Mutable().api_key = StringValue::Get(v); });
	config.AddExtensionOption(
	    "ai_reasoning_effort", "Reasoning effort per request (low/medium/high; empty = omit)", LogicalType::VARCHAR,
	    EnvOr("AI_REASONING_EFFORT", Value("")),
	    [](ClientContext &, SetScope, Value &v) { AIConfig::Mutable().reasoning_effort = StringValue::Get(v); });
	config.AddExtensionOption("ai_concurrency", "Max in-flight LLM calls", LogicalType::UBIGINT,
	                          Value::UBIGINT(ai.max_concurrency), [](ClientContext &, SetScope, Value &v) {
		                          const auto c = v.GetValue<uint64_t>();
		                          if (c == 0) {
			                          throw InvalidInputException("ai_concurrency must be >= 1");
		                          }
		                          AIConfig::Mutable().max_concurrency = c;
	                          });
	config.AddExtensionOption(
	    "ai_max_retries", "Retries per request on throttle (429/503/529) or transient failure", LogicalType::UBIGINT,
	    Value::UBIGINT(ai.max_retries),
	    [](ClientContext &, SetScope, Value &v) { AIConfig::Mutable().max_retries = v.GetValue<uint64_t>(); });
	config.AddExtensionOption(
	    "ai_embed_endpoint", "Embeddings endpoint (text + image)", LogicalType::VARCHAR, Value(ai.embed_url),
	    [](ClientContext &, SetScope, Value &v) { AIConfig::Mutable().embed_url = StringValue::Get(v); });
	config.AddExtensionOption(
	    "ai_embed_model", "Dual-encoder embedding model (text + image)", LogicalType::VARCHAR, Value(ai.embed_model),
	    [](ClientContext &, SetScope, Value &v) { AIConfig::Mutable().embed_model = StringValue::Get(v); });
	config.AddExtensionOption(
	    "ai_embed_images", "Embed image leaves (predicate text x image via the CLIP server) for selectivity",
	    LogicalType::BOOLEAN, Value::BOOLEAN(ai.embed_images),
	    [](ClientContext &, SetScope, Value &v) { AIConfig::Mutable().embed_images = BooleanValue::Get(v); });
	config.AddExtensionOption("ai_embed_concurrency", "Embedding requests in flight at once", LogicalType::UBIGINT,
	                          Value::UBIGINT(ai.embed_concurrency), [](ClientContext &, SetScope, Value &v) {
		                          AIConfig::Mutable().embed_concurrency = MaxValue<idx_t>(v.GetValue<uint64_t>(), 1);
	                          });
	config.AddExtensionOption("ai_embed_batch_images", "Image items per embedding request", LogicalType::UBIGINT,
	                          Value::UBIGINT(ai.embed_batch_images), [](ClientContext &, SetScope, Value &v) {
		                          AIConfig::Mutable().embed_batch_images = MaxValue<idx_t>(v.GetValue<uint64_t>(), 1);
	                          });
	// TypeSafe System One (Jev) as an optional backend for ai_filter (Noul) / ai_classify (Choice)
	config.AddExtensionOption("ai_typesafe",
	                          "Route these AI functions to TypeSafe System One (csv of filter,classify,score)",
	                          LogicalType::VARCHAR,
	                          Value(string(ai.typesafe_filter ? "filter," : "") +
	                                (ai.typesafe_classify ? "classify," : "") + (ai.typesafe_score ? "score" : "")),
	                          [](ClientContext &, SetScope, Value &v) {
		                          const string s = StringValue::Get(v);
		                          AIConfig::Mutable().typesafe_filter = s.find("filter") != string::npos;
		                          AIConfig::Mutable().typesafe_classify = s.find("classify") != string::npos;
		                          AIConfig::Mutable().typesafe_score = s.find("score") != string::npos;
	                          });
	config.AddExtensionOption("ai_typesafe_endpoint", "TypeSafe API base URL (http; the cache proxy terminates TLS)",
	                          LogicalType::VARCHAR, Value(ai.typesafe_url), [](ClientContext &, SetScope, Value &v) {
		                          AIConfig::Mutable().typesafe_url = StringValue::Get(v);
	                          });
	config.AddExtensionOption(
	    "ai_typesafe_model", "TypeSafe model id (jev-latest)", LogicalType::VARCHAR, Value(ai.typesafe_model),
	    [](ClientContext &, SetScope, Value &v) { AIConfig::Mutable().typesafe_model = StringValue::Get(v); });
	config.AddExtensionOption(
	    "ai_typesafe_api_key", "TypeSafe API key", LogicalType::VARCHAR, Value(ai.typesafe_api_key),
	    [](ClientContext &, SetScope, Value &v) { AIConfig::Mutable().typesafe_api_key = StringValue::Get(v); });
	config.AddExtensionOption("ai_typesafe_threshold", "Noul probability at or above which ai_filter is true",
	                          LogicalType::DOUBLE, Value::DOUBLE(ai.typesafe_threshold),
	                          [](ClientContext &, SetScope, Value &v) {
		                          const double t = v.GetValue<double>();
		                          if (t < 0 || t > 1) {
			                          throw InvalidInputException("ai_typesafe_threshold must be in [0, 1]");
		                          }
		                          AIConfig::Mutable().typesafe_threshold = t;
	                          });
	config.AddExtensionOption("ai_local_cache", "Local (in-process) response cache for chat + embeddings",
	                          LogicalType::BOOLEAN, Value::BOOLEAN(true), [](ClientContext &, SetScope, Value &v) {
		                          AIConfig::Mutable().local_cache = BooleanValue::Get(v);
	                          });
	// Optimizer surface (defaults = the soaked SWAN composition)
	config.AddExtensionOption(
	    "ai_factorize", "AI region placement: off/filters/all", LogicalType::VARCHAR,
	    Value(BooleanValue::Get(EnvOnUnlessOff("DUCKDB_AI_DEDUP", true))
	              ? (BooleanValue::Get(EnvOnUnlessOff("DUCKDB_AI_SCAN_REGION", true)) ? "all" : "filters")
	              : "off"));
	{
		// Legacy env mapping: unset -> the new 'factor' default; off/0 -> off; 'factor' -> factor;
		// any other explicit value (the fork's =1) -> 'pushdown' (the behavior its tests assert).
		const char *gj = std::getenv("DUCKDB_AI_GROUP_JOIN");
		const string mode = !gj || !gj[0]                                ? "factor"
		                    : (string(gj) == "off" || string(gj) == "0") ? "off"
		                    : (string(gj) == "factor")                   ? "factor"
		                                                                 : "pushdown";
		config.AddExtensionOption("ai_join_factorize", "AI join strategy: off/pushdown/factor", LogicalType::VARCHAR,
		                          Value(mode));
	}
	config.AddExtensionOption("ai_reorder", "AI predicate reordering + speculative evaluation", LogicalType::BOOLEAN,
	                          EnvOnUnlessOff("DUCKDB_AI_REORDER", true));
	config.AddExtensionOption("ai_pullup", "Semantic filter pull-up above joins", LogicalType::BOOLEAN,
	                          EnvOnUnlessOff("DUCKDB_SEMANTIC_PULLUP", true));
	config.AddExtensionOption("ai_speculative",
	                          "Speculative pre-filter at a pulled-up predicate's leaf (rows the model expects to "
	                          "fail are pruned before the join)",
	                          LogicalType::BOOLEAN, EnvOnUnlessOff("AI_SPECULATIVE", true));
	config.AddExtensionOption("ai_limit", "LIMIT pushdown into AI evaluation", LogicalType::BOOLEAN,
	                          EnvOnUnlessOff("DUCKDB_AI_LIMIT", true));
	config.AddExtensionOption("ai_semi_reduce", "Yannakakis semi-join reduction before AI evaluation",
	                          LogicalType::BOOLEAN, EnvOnUnlessOff("DUCKDB_YANNAKAKIS", true));
	config.AddExtensionOption(
	    "ai_prefix_cache", "Explicit provider prompt caching for factor-graph pair prompts (GPT-5.6+)",
	    LogicalType::BOOLEAN,
	    Value::BOOLEAN(!(std::getenv("AI_PREFIX_CACHE") != nullptr && string(std::getenv("AI_PREFIX_CACHE")) == "off")),
	    [](ClientContext &, SetScope, Value &v) { AIConfig::Mutable().prefix_cache = BooleanValue::Get(v); });
	// ON by default: a materialised CTE is an optimisation barrier, and a semantic filter sealed
	// inside one runs on its FULL base table however selective the outer query is (agent_bench
	// Q22: 89 calls where 10 rows survive; Q26: 1,425 where 5 do). Inlining AI-bearing CTEs is
	// what lets relational pushdown and the semi-join reduction reach the predicate -- it is
	// boundary removal, not an optimisation in itself. Correct since the AIKeyBind idempotence
	// fix (a duplicated AI subtree used to come back mis-bound). See DESIGN.md §5.
	config.AddExtensionOption("ai_inline_ai_ctes",
	                          "Inline CTEs containing AI functions so relational pruning and the "
	                          "semantic pull-up can reach the predicate",
	                          LogicalType::BOOLEAN, EnvOnUnlessOff("AI_INLINE_AI_CTES", true));
	config.AddExtensionOption("ai_hedge", "Hedge straggler calls past the observed p99 latency", LogicalType::BOOLEAN,
	                          EnvOnUnlessOff("AI_HEDGE", true), [](ClientContext &, SetScope, Value &v) {
		                          AIConfig::Mutable().hedge = BooleanValue::Get(v);
	                          });
	config.AddExtensionOption("ai_local_cache_scope",
	                          "Local response cache scope: query (Q1 never serves Q2) or cross_query "
	                          "(process-lifetime, lotus-style)",
	                          LogicalType::VARCHAR, EnvOr("AI_LOCAL_CACHE_SCOPE", Value("query")),
	                          [](ClientContext &, SetScope, Value &v) {
		                          AIConfig::Mutable().local_cache_cross_query = StringValue::Get(v) == "cross_query";
	                          });
	config.AddExtensionOption(
	    "ai_debug_log", "Debug logging channels (csv: region,spec,yann)", LogicalType::VARCHAR, Value(""),
	    [](ClientContext &, SetScope, Value &v) { AIConfig::Mutable().debug_log = StringValue::Get(v); });
	config.AddExtensionOption("ai_debug_prompt_variant", "ai_filter prompt variant: strict/soft/plain",
	                          LogicalType::VARCHAR, EnvOr("AI_PROMPT_VARIANT", Value("strict")),
	                          [](ClientContext &, SetScope, Value &v) {
		                          const string pv = StringValue::Get(v);
		                          AIConfig::Mutable().prompt_variant_soft = pv == "soft";
		                          AIConfig::Mutable().prompt_variant_plain = pv == "plain";
	                          });
	if (const char *pv = std::getenv("AI_PROMPT_VARIANT")) {
		AIConfig::Mutable().prompt_variant_soft = string(pv) == "soft";
		AIConfig::Mutable().prompt_variant_plain = string(pv) == "plain";
	}
	if (const char *cs = std::getenv("AI_LOCAL_CACHE_SCOPE")) {
		AIConfig::Mutable().local_cache_cross_query = string(cs) == "cross_query";
	}
}

static void LoadInternal(ExtensionLoader &loader) {
#ifndef _WIN32
	// The LLM/embedding client reuses keep-alive sockets; on macOS httplib writes without MSG_NOSIGNAL,
	// so a peer that closed an idle pooled connection would otherwise kill the whole process with
	// SIGPIPE (a silent exit 141 mid-query). Ignored, the write fails with EPIPE, which the client
	// already treats as a transport error and retries on a fresh connection.
	signal(SIGPIPE, SIG_IGN);
#endif
	AIInstallCrashReporter();
	RegisterAISettings(loader.GetDatabaseInstance());
	// Stage 1 smoke function; replaced by the full AI function registration as porting proceeds.
	auto version_fn = ScalarFunction("aisql_version", {}, LogicalType::VARCHAR,
	                                 [](DataChunk &args, ExpressionState &state, Vector &result) {
		                                 result.Reference(Value("aisql 0.1.0"), count_t(args.size()));
	                                 });
	loader.RegisterFunction(version_fn);
	RegisterAIFunctions(loader);
	RegisterAIMockFunctions(loader);
	RegisterAisqlOptimizer(loader.GetDatabaseInstance());
}

void AisqlExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string AisqlExtension::Name() {
	return "aisql";
}

std::string AisqlExtension::Version() const {
#ifdef EXT_VERSION_AISQL
	return EXT_VERSION_AISQL;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(aisql, loader) {
	duckdb::LoadInternal(loader);
}
}
