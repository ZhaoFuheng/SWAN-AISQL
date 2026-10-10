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
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/setting_info.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <csignal>
#include <functional>
#include <utility>
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

//! One AIConfig-backed setting: how a SET value lands in an AIConfig. The same table serves the process-wide
//! defaults (registration, `SET GLOBAL`) and the per-connection view (AIConfigForContext): a plain `SET` is
//! session-scoped in DuckDB, so it must change this connection's requests only -- the callback leaves the
//! process-global AIConfig alone for it and the connection's value is read back when its queries run.
struct AIConfigSetting {
	const char *name;
	const char *description;
	LogicalType type;
	Value default_value;
	std::function<void(AIConfig &, const Value &)> apply;
	//! True for the settings that identify WHERE a connection's requests go and how they are priced: a
	//! session SET of one of these changes this connection only. The rest tune process-wide machinery
	//! (the request pool, the embeddings server, caches, logging) and a session SET changes the process.
	bool per_connection = false;
};

static const vector<AIConfigSetting> &AIConfigSettings() {
	static const vector<AIConfigSetting> table = [] {
		const auto &ai = AIConfig::Get(); // force env seeding before defaults are read
		vector<AIConfigSetting> t;
		auto str = [](string AIConfig::*field) {
			return [field](AIConfig &c, const Value &v) {
				c.*field = StringValue::Get(v);
			};
		};
		auto flag = [](bool AIConfig::*field) {
			return [field](AIConfig &c, const Value &v) {
				c.*field = BooleanValue::Get(v);
			};
		};
		auto price = [](double AIConfig::*field) {
			return [field](AIConfig &c, const Value &v) {
				c.*field = v.GetValue<double>();
			};
		};
		t.push_back({"ai_endpoint", "OpenAI-compatible LLM endpoint", LogicalType::VARCHAR, Value(ai.base_url),
		             str(&AIConfig::base_url), true});
		t.push_back({"ai_model", "LLM model id", LogicalType::VARCHAR, Value(ai.model), str(&AIConfig::model), true});
		t.push_back(
		    {"ai_api_key", "LLM API key", LogicalType::VARCHAR, Value(ai.api_key), str(&AIConfig::api_key), true});
		t.push_back({"ai_reasoning_effort", "Reasoning effort per request (low/medium/high; empty = omit)",
		             LogicalType::VARCHAR, EnvOr("AI_REASONING_EFFORT", Value("")), str(&AIConfig::reasoning_effort),
		             true});
		t.push_back({"ai_concurrency", "Max in-flight LLM calls", LogicalType::UBIGINT,
		             Value::UBIGINT(ai.max_concurrency), [](AIConfig &c, const Value &v) {
			             const auto n = v.GetValue<uint64_t>();
			             if (n == 0) {
				             throw InvalidInputException("ai_concurrency must be >= 1");
			             }
			             c.max_concurrency = n;
		             }});
		t.push_back({"ai_max_retries", "Retries per request on throttle (429/503/529) or transient failure",
		             LogicalType::UBIGINT, Value::UBIGINT(ai.max_retries),
		             [](AIConfig &c, const Value &v) { c.max_retries = v.GetValue<uint64_t>(); }, true});
		// Cost accounting for a direct endpoint (the litellm proxy reports the cost per response instead)
		t.push_back({"ai_price_input",
		             "USD per 1M input tokens, for ai_usage().cost_usd on an endpoint that reports no cost",
		             LogicalType::DOUBLE, Value::DOUBLE(ai.price_input_per_mtok),
		             price(&AIConfig::price_input_per_mtok), true});
		t.push_back({"ai_price_output", "USD per 1M output tokens (see ai_price_input)", LogicalType::DOUBLE,
		             Value::DOUBLE(ai.price_output_per_mtok), price(&AIConfig::price_output_per_mtok), true});
		t.push_back({"ai_price_cached", "USD per 1M cached input tokens (see ai_price_input)", LogicalType::DOUBLE,
		             Value::DOUBLE(ai.price_cached_per_mtok), price(&AIConfig::price_cached_per_mtok), true});
		t.push_back({"ai_ca_cert_file", "CA bundle for https endpoints (empty = the system certificate store)",
		             LogicalType::VARCHAR, Value(ai.ca_cert_file), str(&AIConfig::ca_cert_file), true});
		t.push_back({"ai_tls_verify", "Verify the server certificate of https endpoints", LogicalType::BOOLEAN,
		             Value::BOOLEAN(ai.tls_verify), flag(&AIConfig::tls_verify), true});
		t.push_back({"ai_embed_endpoint", "Embeddings endpoint (text + image)", LogicalType::VARCHAR,
		             Value(ai.embed_url), [](AIConfig &c, const Value &v) {
			             c.embed_url = StringValue::Get(v);
			             AIEmbedEndpointReset();
		             }});
		t.push_back({"ai_embed_model", "Dual-encoder embedding model (text + image)", LogicalType::VARCHAR,
		             Value(ai.embed_model), str(&AIConfig::embed_model)});
		t.push_back({"ai_embed_images",
		             "Embed image leaves (predicate text x image via the CLIP server) for selectivity",
		             LogicalType::BOOLEAN, Value::BOOLEAN(ai.embed_images), flag(&AIConfig::embed_images)});
		t.push_back({"ai_embed_concurrency", "Embedding requests in flight at once", LogicalType::UBIGINT,
		             Value::UBIGINT(ai.embed_concurrency), [](AIConfig &c, const Value &v) {
			             c.embed_concurrency = MaxValue<idx_t>(v.GetValue<uint64_t>(), 1);
		             }});
		t.push_back({"ai_embed_batch_images", "Image items per embedding request", LogicalType::UBIGINT,
		             Value::UBIGINT(ai.embed_batch_images), [](AIConfig &c, const Value &v) {
			             c.embed_batch_images = MaxValue<idx_t>(v.GetValue<uint64_t>(), 1);
		             }});
		// TypeSafe System One (Jev) as an optional backend for ai_filter (Noul) / ai_classify (Choice)
		t.push_back({"ai_typesafe", "Route these AI functions to TypeSafe System One (csv of filter,classify,score)",
		             LogicalType::VARCHAR,
		             Value(string(ai.typesafe_filter ? "filter," : "") + (ai.typesafe_classify ? "classify," : "") +
		                   (ai.typesafe_score ? "score" : "")),
		             [](AIConfig &c, const Value &v) {
			             const string s = StringValue::Get(v);
			             c.typesafe_filter = s.find("filter") != string::npos;
			             c.typesafe_classify = s.find("classify") != string::npos;
			             c.typesafe_score = s.find("score") != string::npos;
		             }});
		t.push_back({"ai_typesafe_endpoint", "TypeSafe API base URL (http; the cache proxy terminates TLS)",
		             LogicalType::VARCHAR, Value(ai.typesafe_url), str(&AIConfig::typesafe_url)});
		t.push_back({"ai_typesafe_model", "TypeSafe model id (jev-latest)", LogicalType::VARCHAR,
		             Value(ai.typesafe_model), str(&AIConfig::typesafe_model)});
		t.push_back({"ai_typesafe_api_key", "TypeSafe API key", LogicalType::VARCHAR, Value(ai.typesafe_api_key),
		             str(&AIConfig::typesafe_api_key)});
		t.push_back({"ai_typesafe_threshold", "Noul probability at or above which ai_filter is true",
		             LogicalType::DOUBLE, Value::DOUBLE(ai.typesafe_threshold), [](AIConfig &c, const Value &v) {
			             const double th = v.GetValue<double>();
			             if (th < 0 || th > 1) {
				             throw InvalidInputException("ai_typesafe_threshold must be in [0, 1]");
			             }
			             c.typesafe_threshold = th;
		             }});
		t.push_back({"ai_local_cache", "Local (in-process) response cache for chat + embeddings", LogicalType::BOOLEAN,
		             Value::BOOLEAN(true), flag(&AIConfig::local_cache)});
		t.push_back({"ai_prefix_cache", "Explicit provider prompt caching for factor-graph pair prompts (GPT-5.6+)",
		             LogicalType::BOOLEAN,
		             Value::BOOLEAN(!(std::getenv("AI_PREFIX_CACHE") != nullptr &&
		                              string(std::getenv("AI_PREFIX_CACHE")) == "off")),
		             flag(&AIConfig::prefix_cache)});
		t.push_back({"ai_hedge", "Hedge straggler calls past the observed p99 latency", LogicalType::BOOLEAN,
		             EnvOnUnlessOff("AI_HEDGE", true), flag(&AIConfig::hedge)});
		t.push_back(
		    {"ai_local_cache_scope",
		     "Local response cache scope: query (Q1 never serves Q2) or cross_query (process-lifetime, lotus-style)",
		     LogicalType::VARCHAR, EnvOr("AI_LOCAL_CACHE_SCOPE", Value("query")), [](AIConfig &c, const Value &v) {
			     c.local_cache_cross_query = StringValue::Get(v) == "cross_query";
		     }});
		t.push_back({"ai_debug_log", "Debug logging channels (csv: region,spec,yann)", LogicalType::VARCHAR, Value(""),
		             str(&AIConfig::debug_log)});
		t.push_back({"ai_debug_prompt_variant", "ai_filter prompt variant: strict/soft/plain", LogicalType::VARCHAR,
		             EnvOr("AI_PROMPT_VARIANT", Value("strict")), [](AIConfig &c, const Value &v) {
			             const string pv = StringValue::Get(v);
			             c.prompt_variant_soft = pv == "soft";
			             c.prompt_variant_plain = pv == "plain";
		             }});
		return t;
	}();
	return table;
}

//! This connection's view of the client configuration: the process-wide AIConfig (environment seeds, `SET
//! GLOBAL`, the mock) with every AIConfig-backed setting this connection has SET in session scope applied on
//! top. Built per request batch; a few dozen setting lookups.
AIConfig AIConfigForContext(ClientContext &context) {
	AIConfig config = AIConfig::Get();
	auto &db_config = DBConfig::GetConfig(context);
	for (auto &setting : AIConfigSettings()) {
		if (!setting.per_connection) {
			continue; // process-wide: a session SET already went into AIConfig::Mutable()
		}
		optional_ptr<const ConfigurationOption> option;
		const auto index = db_config.TryGetSettingIndex(Identifier(setting.name), option);
		if (!index.IsValid()) {
			continue;
		}
		Value value;
		auto found = context.TryGetCurrentUserSetting(index.GetIndex(), value);
		if (found && found.GetScope() == SettingScope::LOCAL) {
			setting.apply(config, value);
		}
	}
	return config;
}

// DuckDB takes a plain function pointer as the SET callback (no captures), so each table entry gets its own
// instantiation that finds its row by index.
constexpr size_t kAIConfigSettingCount = 27;

template <size_t I>
static void AIConfigSettingCallback(ClientContext &, SetScope scope, Value &v) {
	const auto &setting = AIConfigSettings()[I];
	if (setting.per_connection && scope != SetScope::GLOBAL) {
		AIConfig trial = AIConfig::Get();
		setting.apply(trial, v); // validate (a bad value throws); the connection reads the value back itself
		return;
	}
	setting.apply(AIConfig::Mutable(), v);
}

template <size_t... Is>
static void RegisterAIConfigSettings(DBConfig &config, std::index_sequence<Is...>) {
	const auto &table = AIConfigSettings();
	if (table.size() != kAIConfigSettingCount) {
		throw InternalException("aisql: %llu AIConfig settings in the table, kAIConfigSettingCount is %llu",
		                        static_cast<uint64_t>(table.size()), static_cast<uint64_t>(kAIConfigSettingCount));
	}
	(config.AddExtensionOption(table[Is].name, table[Is].description, table[Is].type, table[Is].default_value,
	                           &AIConfigSettingCallback<Is>),
	 ...);
}

//! Registers the settings. The AIConfig-backed ones come from the table: a `SET GLOBAL` changes the process
//! default, a session `SET` is validated here and read back per connection (AIConfigForContext). The
//! optimizer flags are read from the connection by the optimizer and need no bridge.
static void RegisterAISettings(DatabaseInstance &db) {
	auto &config = DBConfig::GetConfig(db);
	RegisterAIConfigSettings(config, std::make_index_sequence<kAIConfigSettingCount> {});
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
	config.AddExtensionOption("ai_factor_state",
	                          "Factor graph scheduler state: dense (one byte per pair of each edge) or sparse "
	                          "(counters per member and the pairs asked; no pair-domain allocation)",
	                          LogicalType::VARCHAR, EnvOr("AI_FACTOR_STATE", Value("sparse")));
	config.AddExtensionOption("ai_factor_max_terms",
	                          "Largest disjunctive normal form (terms) the factor graph takes for a join predicate "
	                          "that is not a conjunction of factors; beyond it the join runs as a region",
	                          LogicalType::UBIGINT, Value::UBIGINT(16));
	config.AddExtensionOption("ai_factor_pair_limit",
	                          "Largest estimated pair domain (left rows x right rows of one AI join edge) the factor "
	                          "graph takes under the dense scheduler or for a cyclic edge graph; above it the join is "
	                          "evaluated as a region over the cross product",
	                          LogicalType::UBIGINT, Value::UBIGINT(100000000ULL));
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
