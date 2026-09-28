//===----------------------------------------------------------------------===//
// In-process deterministic mock LLM backend for tests.
//
// CALL ai_mock_start() spins a duckdb_httplib server on an ephemeral port serving
// /v1/chat/completions and /v1/embeddings with answers that are a stable md5 hash of the prompt
// (semantics ported 1:1 from the fork's test/ai-sql/mock_backend.py, including python's
// json.dumps(sort_keys=True) seed for multimodal content), and points the live AIConfig at it.
// CALL ai_mock_stop() shuts it down and restores the previous endpoints. Starting the mock also resets
// the process-global AI config to its environment defaults, clears the in-process response/embedding
// caches and the selectivity model: a SET writes a process-wide value through its callback, so without
// this a setting, a cached answer or a trained model left by one test file leaked into the next file
// run in the same unittest process (CI batches ten files per process).
//===----------------------------------------------------------------------===//
#include "ai_mock_server.hpp"

#include "ai_client.hpp"
#include "ai_selectivity_model.hpp"
#include "duckdb/common/crypto/md5.hpp"
#include "duckdb/function/table_function.hpp"
#include "yyjson.hpp"

#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include <thread>

#include "httplib.hpp"

namespace duckdb {

using namespace duckdb_yyjson; // NOLINT

namespace {

//! Python `int(md5(s).hexdigest(), 16)` is the digest read as one big-endian 128-bit integer.
struct MockHash {
	uint8_t digest[MD5Context::MD5_HASH_LENGTH_BINARY];

	explicit MockHash(const string &s) {
		MD5Context ctx;
		ctx.Add(s);
		ctx.Finish(digest);
	}
	//! h(s) % m, big-endian byte-wise modular reduction (exact for any m > 0).
	uint64_t Mod(uint64_t m) const {
		uint64_t r = 0;
		for (auto b : digest) {
			r = (r * 256 + b) % m;
		}
		return r;
	}
	//! (h(s) >> (8*j)) & 0xFF — the j-th byte from the little end of the big-endian integer.
	uint8_t ByteFromLow(idx_t j) const {
		return digest[MD5Context::MD5_HASH_LENGTH_BINARY - 1 - j];
	}
};

//! Serialize a yyjson value exactly like python json.dumps(v, sort_keys=True): ", " / ": "
//! separators, sorted object keys. Only needs the shapes our client sends (obj/arr/str/num/bool).
void PyDumps(yyjson_val *val, string &out) {
	if (!val) {
		out += "null";
		return;
	}
	if (yyjson_is_obj(val)) {
		vector<std::pair<string, yyjson_val *>> entries;
		size_t idx, max;
		yyjson_val *key, *v;
		yyjson_obj_foreach(val, idx, max, key, v) {
			entries.emplace_back(string(yyjson_get_str(key), yyjson_get_len(key)), v);
		}
		std::sort(entries.begin(), entries.end(),
		          [](const std::pair<string, yyjson_val *> &a, const std::pair<string, yyjson_val *> &b) {
			          return a.first < b.first;
		          });
		out += "{";
		for (idx_t i = 0; i < entries.size(); i++) {
			if (i > 0) {
				out += ", ";
			}
			out += "\"" + entries[i].first + "\": ";
			PyDumps(entries[i].second, out);
		}
		out += "}";
		return;
	}
	if (yyjson_is_arr(val)) {
		out += "[";
		size_t idx, max;
		yyjson_val *v;
		yyjson_arr_foreach(val, idx, max, v) {
			if (idx > 0) {
				out += ", ";
			}
			PyDumps(v, out);
		}
		out += "]";
		return;
	}
	if (yyjson_is_str(val)) {
		// python json.dumps escapes like JSON with ensure_ascii; our content is ASCII-safe in
		// practice (URLs/base64/prompt text) — escape the JSON specials.
		out += "\"";
		const char *s = yyjson_get_str(val);
		size_t n = yyjson_get_len(val);
		for (size_t i = 0; i < n; i++) {
			unsigned char c = static_cast<unsigned char>(s[i]);
			switch (c) {
			case '"':
				out += "\\\"";
				break;
			case '\\':
				out += "\\\\";
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
		out += "\"";
		return;
	}
	if (yyjson_is_bool(val)) {
		out += yyjson_get_bool(val) ? "true" : "false";
		return;
	}
	if (yyjson_is_int(val)) {
		out += std::to_string(yyjson_get_sint(val));
		return;
	}
	if (yyjson_is_real(val)) {
		char buf[32];
		snprintf(buf, sizeof(buf), "%g", yyjson_get_real(val));
		out += buf;
		return;
	}
	out += "null";
}

//! Build the JSON-schema-driven mock answer: same field rules as the python mock.
string GenFromSchema(yyjson_val *schema, const string &seed) {
	string out = "{";
	bool first = true;
	yyjson_val *props = schema ? yyjson_obj_get(schema, "properties") : nullptr;
	if (props) {
		size_t idx, max;
		yyjson_val *key, *spec;
		yyjson_obj_foreach(props, idx, max, key, spec) {
			const string name(yyjson_get_str(key), yyjson_get_len(key));
			if (!first) {
				out += ", ";
			}
			first = false;
			out += "\"" + name + "\": ";
			MockHash h(seed + name);
			yyjson_val *enum_vals = yyjson_obj_get(spec, "enum");
			yyjson_val *type_val = yyjson_obj_get(spec, "type");
			const string type = type_val && yyjson_is_str(type_val) ? yyjson_get_str(type_val) : "";
			if (enum_vals && yyjson_is_arr(enum_vals) && yyjson_arr_size(enum_vals) > 0) {
				auto pick = h.Mod(yyjson_arr_size(enum_vals));
				yyjson_val *v = yyjson_arr_get(enum_vals, pick);
				string tmp;
				PyDumps(v, tmp);
				out += tmp;
			} else if (type == "boolean") {
				// test knob: '[p=NN]' in the seed sets the pass PERCENTAGE (deterministic per
				// prompt: hash mod 100 < NN); absent -> the historic 50/50 coin. The LAST marker
				// wins, so a marker embedded in a row VALUE overrides the predicate's default --
				// that is how benches build per-member heterogeneous selectivities.
				const auto p_pos = seed.rfind("[p=");
				if (p_pos != string::npos) {
					const int pct = std::atoi(seed.c_str() + p_pos + 3);
					out += (h.Mod(100) < static_cast<uint64_t>(pct < 0 ? 0 : pct)) ? "true" : "false";
				} else {
					out += (h.Mod(2) == 0) ? "true" : "false";
				}
			} else if (type == "number" || type == "integer") {
				out += std::to_string(h.Mod(100));
			} else {
				out += "\"x\"";
			}
		}
	}
	out += "}";
	return out;
}

struct MockServerState {
	std::unique_ptr<duckdb_httplib::Server> server;
	//! The listener thread. Joined in AIMockStop BEFORE the server is destroyed: is_running() turns false
	//! while listen_after_bind() is still tearing down its task queue, so destroying the server on that
	//! signal alone freed it under the listener (a silent crash in the next test file on Windows).
	std::thread listener;
	int port = 0;
	// endpoints to restore on stop
	string saved_base_url;
	string saved_embed_url;
	string saved_typesafe_url;
	string saved_model;
};

static std::mutex &MockMutex() {
	static std::mutex value;
	return value;
}
//! Requests that carried an explicit prompt-cache breakpoint (write or read), for policy tests.
static std::atomic<int64_t> &MockBreakpointRequests() {
	static std::atomic<int64_t> value {0};
	return value;
}
// Leaked on purpose: a detached listener thread may outlive main() (a failed test skips
// ai_mock_stop), and static destruction of the server while it runs would crash the harness.
static MockServerState &Mock() {
	static MockServerState &state = *(new MockServerState());
	return state;
}

void HandlePost(const duckdb_httplib::Request &req, duckdb_httplib::Response &res) {
	yyjson_doc *doc = yyjson_read(req.body.c_str(), req.body.size(), 0);
	yyjson_val *root = doc ? yyjson_doc_get_root(doc) : nullptr;
	// PROVIDER FIDELITY: a body that is not valid JSON is a 400, not a benign answer. Silently
	// answering "ok" hid a real bug -- a mis-bound json_schema embedded a raw column value, which
	// the provider rejected but the mock happily served (see keybind_idempotent.test).
	if (!doc && !req.body.empty()) {
		res.status = 400;
		res.set_content("{\"error\":{\"message\":\"mock: request body is not valid JSON\","
		                "\"type\":\"invalid_request_error\"}}",
		                "application/json");
		return;
	}
	string body;
	if (StringUtil::EndsWith(req.path, "/systemone")) {
		// TypeSafe System One: one question "q" over `state`. Seeded like the chat path's boolean
		// schema (state + "result", '[p=NN]' knob) so a prompt gets the SAME verdict from either
		// backend; a choice picks by hash among the criteria keys.
		yyjson_val *state = root ? yyjson_obj_get(root, "state") : nullptr;
		string seed = state && yyjson_is_str(state) ? string(yyjson_get_str(state), yyjson_get_len(state)) : "";
		yyjson_val *questions = root ? yyjson_obj_get(root, "questions") : nullptr;
		yyjson_val *q = questions ? yyjson_obj_get(questions, "q") : nullptr;
		yyjson_val *type = q ? yyjson_obj_get(q, "type") : nullptr;
		const string qtype = type && yyjson_is_str(type) ? yyjson_get_str(type) : "";
		yyjson_val *criteria = q ? yyjson_obj_get(q, "criteria") : nullptr;
		if (qtype == "noul") {
			MockHash h(seed + "result");
			const auto p_pos = seed.rfind("[p=");
			const bool yes = p_pos != string::npos
			                     ? h.Mod(100) < static_cast<uint64_t>(std::max(0, std::atoi(seed.c_str() + p_pos + 3)))
			                     : h.Mod(2) == 0;
			body = string("{\"model\":\"mock\",\"answers\":{\"q\":{\"type\":\"noul\",\"noul\":") +
			       (yes ? "0.9" : "0.1") + "}},\"usage\":{\"input_tokens\":8,\"output_tokens\":2}}";
		} else if (qtype == "score" && criteria && yyjson_is_arr(criteria) && yyjson_arr_size(criteria) >= 2) {
			// a Score answers the expected level in [0, levels-1]: seeded by the state, two decimals
			MockHash h(seed + "result");
			const uint64_t levels = yyjson_arr_size(criteria);
			const double score = static_cast<double>(h.Mod(levels * 100 - 99)) / 100.0;
			body = "{\"model\":\"mock\",\"answers\":{\"q\":{\"type\":\"score\",\"score\":" + std::to_string(score) +
			       ",\"confidence\":0.5}},\"usage\":{\"input_tokens\":8,\"output_tokens\":2}}";
		} else if (qtype == "choice" && criteria && yyjson_is_obj(criteria) && yyjson_obj_size(criteria) > 0) {
			vector<string> keys;
			size_t kidx, kmax;
			yyjson_val *k, *v;
			yyjson_obj_foreach(criteria, kidx, kmax, k, v) {
				keys.emplace_back(yyjson_get_str(k), yyjson_get_len(k));
			}
			MockHash h(seed + "result");
			const string &pick = keys[h.Mod(keys.size())];
			body = "{\"model\":\"mock\",\"answers\":{\"q\":{\"type\":\"choice\",\"choice\":\"" + pick +
			       "\",\"confidence\":1.0}},\"usage\":{\"input_tokens\":8,\"output_tokens\":2}}";
		} else {
			res.status = 422;
			res.set_content("{\"error\":{\"message\":\"mock: malformed systemone question\"}}", "application/json");
			if (doc) {
				yyjson_doc_free(doc);
			}
			return;
		}
	} else if (StringUtil::EndsWith(req.path, "/embeddings")) {
		// input: string or array of strings -> 4-dim embeddings from the hash bytes
		yyjson_val *input = root ? yyjson_obj_get(root, "input") : nullptr;
		vector<string> inputs;
		if (input && yyjson_is_str(input)) {
			inputs.emplace_back(yyjson_get_str(input), yyjson_get_len(input));
		} else if (input && yyjson_is_arr(input)) {
			size_t idx, max;
			yyjson_val *v;
			yyjson_arr_foreach(input, idx, max, v) {
				if (yyjson_is_str(v)) {
					inputs.emplace_back(yyjson_get_str(v), yyjson_get_len(v));
				} else {
					string tmp;
					PyDumps(v, tmp);
					inputs.push_back(tmp);
				}
			}
		}
		// test knob: '[esleep=NN]' in an embed input delays this request by NN ms per marked item
		for (auto &in : inputs) {
			const auto p = in.find("[esleep=");
			if (p != string::npos) {
				const int ms = std::atoi(in.c_str() + p + 8);
				if (ms > 0) {
					std::this_thread::sleep_for(std::chrono::milliseconds(ms));
				}
			}
		}
		body = "{\"object\": \"list\", \"data\": [";
		for (idx_t i = 0; i < inputs.size(); i++) {
			MockHash h(inputs[i]);
			if (i > 0) {
				body += ", ";
			}
			body += "{\"object\": \"embedding\", \"index\": " + std::to_string(i) + ", \"embedding\": [";
			for (idx_t j = 0; j < 4; j++) {
				if (j > 0) {
					body += ", ";
				}
				const double v = std::round(h.ByteFromLow(j) / 255.0 * 10000.0) / 10000.0;
				char buf[16];
				snprintf(buf, sizeof(buf), "%g", v);
				body += buf;
			}
			body += "]}";
		}
		body += "], \"usage\": {\"total_tokens\": " + std::to_string(inputs.size()) + "}}";
	} else {
		// chat completion: seed = last user message content (python-dumps'd when multimodal)
		if (req.body.find("\"prompt_cache_breakpoint\"") != string::npos) {
			MockBreakpointRequests().fetch_add(1);
		}
		string seed;
		yyjson_val *msgs = root ? yyjson_obj_get(root, "messages") : nullptr;
		if (msgs && yyjson_is_arr(msgs)) {
			size_t n = yyjson_arr_size(msgs);
			for (size_t i = n; i > 0; i--) {
				yyjson_val *m = yyjson_arr_get(msgs, i - 1);
				yyjson_val *role = m ? yyjson_obj_get(m, "role") : nullptr;
				if (role && yyjson_is_str(role) && string(yyjson_get_str(role)) == "user") {
					yyjson_val *content = yyjson_obj_get(m, "content");
					if (content && yyjson_is_str(content)) {
						seed = string(yyjson_get_str(content), yyjson_get_len(content));
					} else if (content) {
						// Provider fidelity: the SAME visible tokens must give the SAME answer, so a
						// prompt split for explicit caching (text blocks + breakpoint field) hashes
						// exactly like its plain form: an all-text array seeds as the concatenated
						// text. Arrays with images keep the historic PyDumps seed.
						bool all_text = yyjson_is_arr(content);
						string concat;
						if (all_text) {
							size_t pidx, pmax;
							yyjson_val *part;
							yyjson_arr_foreach(content, pidx, pmax, part) {
								yyjson_val *ptype = yyjson_is_obj(part) ? yyjson_obj_get(part, "type") : nullptr;
								yyjson_val *ptext = yyjson_is_obj(part) ? yyjson_obj_get(part, "text") : nullptr;
								if (!ptype || !yyjson_is_str(ptype) || string(yyjson_get_str(ptype)) != "text" ||
								    !ptext || !yyjson_is_str(ptext)) {
									all_text = false;
									break;
								}
								concat += string(yyjson_get_str(ptext), yyjson_get_len(ptext));
							}
						}
						if (all_text) {
							seed = std::move(concat);
						} else {
							PyDumps(content, seed);
						}
					}
					break;
				}
			}
		}
		// test knob: a '[sleep=NN]' marker anywhere in the seed delays this answer by NN ms,
		// letting scheduler tests inject realistic per-prompt latency deterministically.
		const auto sleep_pos = seed.find("[sleep=");
		if (sleep_pos != string::npos) {
			const int ms = std::atoi(seed.c_str() + sleep_pos + 7);
			if (ms > 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds(ms));
			}
		}
		yyjson_val *rf = root ? yyjson_obj_get(root, "response_format") : nullptr;
		string content;
		if (rf) {
			yyjson_val *js = yyjson_obj_get(rf, "json_schema");
			yyjson_val *schema = js ? yyjson_obj_get(js, "schema") : nullptr;
			content = GenFromSchema(schema, seed);
		} else if (seed.find("Return a single yes or no") != string::npos) {
			// A bare prompt that asks for a yes/no gets a yes/no, not "ok": the plain ai_filter
			// variant reads its verdict from the answer TEXT, so a mock that always answered "ok"
			// would make every such predicate unparseable and silently keep zero rows. Same
			// deterministic '[p=NN]' knob as the boolean-schema path, so the two agree per prompt.
			// seeded exactly as the boolean-schema path ("result" is that schema's field name),
			// so a given prompt gets the SAME verdict whether or not the schema was requested
			MockHash h(seed + "result");
			const auto p_pos = seed.rfind("[p=");
			const bool yes = p_pos != string::npos
			                     ? h.Mod(100) < static_cast<uint64_t>(std::max(0, std::atoi(seed.c_str() + p_pos + 3)))
			                     : h.Mod(2) == 0;
			content = yes ? "yes" : "no";
		} else {
			content = "ok";
		}
		// content is embedded as a JSON string value
		string escaped;
		yyjson_mut_doc *tmp_doc = yyjson_mut_doc_new(nullptr);
		yyjson_mut_val *sval = yyjson_mut_strncpy(tmp_doc, content.c_str(), content.size());
		yyjson_mut_doc_set_root(tmp_doc, sval);
		char *sjson = yyjson_mut_write(tmp_doc, 0, nullptr);
		escaped = sjson ? sjson : "\"\"";
		free(sjson);
		yyjson_mut_doc_free(tmp_doc);
		body = "{\"choices\": [{\"message\": {\"content\": " + escaped +
		       "}}], \"usage\": {\"prompt_tokens\": 8, \"completion_tokens\": 2, \"total_tokens\": 10}}";
		if (AIConfig::Get().debug_log.find("mock") != string::npos) {
			fprintf(stderr, "[mock] seed=%s -> %s\n", seed.substr(0, 120).c_str(), content.c_str());
		}
	}
	if (doc) {
		yyjson_doc_free(doc);
	}
	res.set_header("x-litellm-response-cost", "0");
	res.set_content(body, "application/json");
}

} // namespace

int AIMockStart() {
	MockBreakpointRequests().store(0);
	std::lock_guard<std::mutex> lock(MockMutex());
	// A clean slate per test file (see the header comment): reset the process-global client state, and
	// give the file its own server -- a server left running by the previous file (most files never call
	// ai_mock_stop) is stopped and its listener joined first, so no thread of an earlier file survives.
	AIConfig::ResetToDefaults();
	AILocalCacheClear();
	AISelectivityModel::Global().Reset();
	if (Mock().server) {
		Mock().server->stop();
		if (Mock().listener.joinable()) {
			Mock().listener.join();
		}
		Mock().server.reset();
		Mock().port = 0;
	}
	auto server = std::make_unique<duckdb_httplib::Server>();
	// Keep-alive clients hold one server thread per persistent connection; the default pool
	// (~hardware threads) starves under 20+ pooled engine connections, so size it explicitly.
	server->new_task_queue = [] {
		return new duckdb_httplib::ThreadPool(64);
	};
	server->Post(R"(.*)", HandlePost);
	server->Get(R"(.*)", [](const duckdb_httplib::Request &, duckdb_httplib::Response &res) {
		res.set_content("{\"status\": \"ok\"}", "application/json");
	});
	const int port = server->bind_to_any_port("127.0.0.1");
	if (port <= 0) {
		throw IOException("ai_mock_start: could not bind a port");
	}
	auto *srv = server.get();
	Mock().server = std::move(server);
	Mock().port = port;
	Mock().listener = std::thread([srv]() { srv->listen_after_bind(); });
	while (!srv->is_running()) {
		std::this_thread::yield();
	}
	auto &cfg = AIConfig::Mutable();
	Mock().saved_base_url = cfg.base_url;
	Mock().saved_embed_url = cfg.embed_url;
	Mock().saved_typesafe_url = cfg.typesafe_url;
	Mock().saved_model = cfg.model;
	const string url = "http://127.0.0.1:" + std::to_string(port);
	cfg.base_url = url;
	cfg.embed_url = url;
	cfg.typesafe_url = url;
	cfg.model = "mock";
	return port;
}

bool AIMockStop() {
	std::lock_guard<std::mutex> lock(MockMutex());
	if (!Mock().server) {
		return false;
	}
	Mock().server->stop();
	if (Mock().listener.joinable()) {
		Mock().listener.join(); // listen_after_bind() has returned: the task queue is drained and destroyed
	}
	Mock().server.reset();
	Mock().port = 0;
	auto &cfg = AIConfig::Mutable();
	cfg.base_url = Mock().saved_base_url;
	cfg.embed_url = Mock().saved_embed_url;
	cfg.typesafe_url = Mock().saved_typesafe_url;
	cfg.model = Mock().saved_model;
	return true;
}

//===--------------------------------------------------------------------===//
// Table functions: CALL ai_mock_start() / CALL ai_mock_stop()
//===--------------------------------------------------------------------===//
struct AIMockBindData : public TableFunctionData {
	bool done = false;
};

static unique_ptr<FunctionData> AIMockBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &return_types,
                                           vector<Identifier> &names) {
	return_types.emplace_back(LogicalType::INTEGER);
	names.emplace_back("port");
	return make_uniq<AIMockBindData>();
}

static void AIMockStartFunction(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->CastNoConst<AIMockBindData>();
	if (bind.done) {
		return;
	}
	bind.done = true;
	output.SetChildCardinality(1);
	output.data[0].SetValue(0, Value::INTEGER(AIMockStart()));
}

static void AIMockStopFunction(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->CastNoConst<AIMockBindData>();
	if (bind.done) {
		return;
	}
	bind.done = true;
	output.SetChildCardinality(1);
	output.data[0].SetValue(0, Value::INTEGER(AIMockStop() ? 1 : 0));
}

static unique_ptr<FunctionData> AIMockBreakpointsBind(ClientContext &, TableFunctionBindInput &,
                                                      vector<LogicalType> &return_types, vector<Identifier> &names) {
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("breakpoint_requests");
	return make_uniq<AIMockBindData>();
}

static void AIMockBreakpointsFunction(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &bind = data.bind_data->CastNoConst<AIMockBindData>();
	if (bind.done) {
		return;
	}
	bind.done = true;
	output.SetChildCardinality(1);
	output.data[0].SetValue(0, Value::BIGINT(MockBreakpointRequests().load()));
}

void RegisterAIMockFunctions(ExtensionLoader &loader) {
	loader.RegisterFunction(TableFunction("ai_mock_start", {}, AIMockStartFunction, AIMockBind));
	loader.RegisterFunction(TableFunction("ai_mock_stop", {}, AIMockStopFunction, AIMockBind));
	loader.RegisterFunction(TableFunction("ai_mock_breakpoints", {}, AIMockBreakpointsFunction, AIMockBreakpointsBind));
}

} // namespace duckdb
