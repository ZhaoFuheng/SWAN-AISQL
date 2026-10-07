#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/vector/unified_vector_format.hpp"
#include "duckdb/common/vector/list_vector.hpp"
#include "ai_client.hpp"
#include "ai_filter_selectivity.hpp"
#include "ai_prompt_cost.hpp"
#include "ai_selectivity_model.hpp"
#include "filter_tree_order.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/allocator.hpp"
#include "ai_dedup.hpp"
#include "optimizer/ai_filter_tree_build.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace duckdb {

//===--------------------------------------------------------------------===//
// JSON schema fragments (structured output)
//===--------------------------------------------------------------------===//
// Fixed structured-output schemas shared by all AI functions except ai_complete.
// Every function returns a "result" field; only the leaf type differs (ai_filter adds its reasoning).
static const char *const NUMBER_RESULT_SCHEMA =
    "{\"type\":\"object\",\"properties\":{\"result\":{\"type\":\"number\"}},"
    "\"required\":[\"result\"],\"additionalProperties\":false}";
static const char *const STRING_RESULT_SCHEMA =
    "{\"type\":\"object\",\"properties\":{\"result\":{\"type\":\"string\"}},"
    "\"required\":[\"result\"],\"additionalProperties\":false}";
//! ai_filter answers with a short "reasoning" before its "result": a reasoning model commits to a bare
//! boolean with a much smaller thinking budget than it spends on a free-text verdict (SWAN 2.0 bench:
//! ~15 vs ~45 reasoning tokens per call for the same claims), and the sentence recovers that accuracy.
static const char *const FILTER_RESULT_SCHEMA =
    "{\"type\":\"object\",\"properties\":{\"reasoning\":{\"type\":\"string\",\"description\":\"one short "
    "sentence\"},\"result\":{\"type\":\"boolean\"}},\"required\":[\"reasoning\",\"result\"],"
    "\"additionalProperties\":false}";

//===--------------------------------------------------------------------===//
// System prompts, shared by the scalar functions AND the reorder node's leaf evaluation (AIEvalLeaf) so both
// paths stay byte-identical (result-preserving reorder + shared cache keys). The framing follows LOTUS's
// semantic-operator templates (lotus/templates/task_instructions.py, sem_agg.py): describe the user's input
// and the job, rather than a persona ("You are a precise ..."), which measurably yes-biases ambiguous
// filters. The answer FORMAT stays SWAN's structured-output JSON schema, so LOTUS's "Answer: True/False"
// format instructions are intentionally omitted.
static const char *const AI_FILTER_SYSTEM_PROMPT =
    "The user will provide a claim and some relevant context.\n"
    "Your job is to determine whether the claim is true for the given context.";
//! Softer variant (SET ai_debug_prompt_variant='soft'): the strict wording judges borderline
//! cases conservatively (it cost quality on the MOVIE bench's clearly-positive filters); the soft
//! wording asks for the natural reading instead of strict entailment.
static const char *const AI_FILTER_SYSTEM_PROMPT_SOFT =
    "The user will provide a statement and some relevant context.\n"
    "Decide whether the statement holds for the given context, using the natural reading a "
    "careful person would apply. Lean towards the answer an attentive human reviewer would give.";
//! Returns the active ai_filter system prompt per ai_debug_prompt_variant.
static const char *AIFilterSystemPrompt() {
	if (AIConfig::Get().prompt_variant_plain) {
		return ""; // bare prompt: the caller's own text carries the answer-format instruction
	}
	return AIConfig::Get().prompt_variant_soft ? AI_FILTER_SYSTEM_PROMPT_SOFT : AI_FILTER_SYSTEM_PROMPT;
}

//! ai_filter's structured-output schema, or "" in the plain variant (bare prompt, text verdict).
static const char *AIFilterSchema() {
	return AIConfig::Get().prompt_variant_plain ? "" : FILTER_RESULT_SCHEMA;
}

//! Read ai_filter's verdict: the {"result":bool} envelope, else a bare yes/no answer text. The
//! text fallback is what lets a plain-variant call consume an answer produced without the schema.
static bool AIReadFilterVerdict(const string &content, bool &out) {
	return AIParseBoolField(content, "result", out) || AIParseVerdictText(content, out);
}
//! TypeSafe System One shapes (used only when SET ai_typesafe routes the function there; the
//! chat prompt/system/schema above stay set, so the chat path and cache key are unchanged).
//! ai_filter -> a Noul over the whole prompt (the state stays byte-identical to the chat prompt).
//! The instruction restates the system prompt's job and, when known, QUOTES the predicate text --
//! the prompt's constant part (AISplitPrompt), the same split the selectivity features use. Jev
//! reads literally: over one undifferentiated blob it compresses every answer towards 0.5
//! (ECOMM q1: perfect ranking, nothing above 0.48); naming the claim separates the two modes
//! (0.95+ vs <0.5) while a whole-prompt state keeps pair prompts' "Review 1/Review 2" labels,
//! which a state reduced to the row-varying text would lose.
static void AIMarkFilterQuestion(AIRequest &req, const string &pred_text) {
	req.question = AIRequest::Question::NOUL;
	req.instructions = "The state contains a claim together with the context it refers to. ";
	if (!pred_text.empty()) {
		req.instructions += "The claim is the fixed part of the state: \"" + pred_text +
		                    "\" (its blank slots are filled by the context). ";
	}
	req.instructions += "Is the claim true for that context?";
}
//! ai_classify -> a Choice over the listed categories, with the classified input as the state.
static void AIMarkClassifyQuestion(AIRequest &req, string input, vector<std::pair<string, string>> options,
                                   const string &extra_instruction) {
	req.question = AIRequest::Question::CHOICE;
	req.state = std::move(input);
	req.options = std::move(options);
	req.instructions = "Which of the listed categories best matches the state?";
	if (!extra_instruction.empty()) {
		req.instructions += " " + extra_instruction;
	}
}
//! The description a rubric gives level `n`, when it lists levels as "n: text", "n. text", "n) text" or
//! "n - text" lines (SemBench MOVIE writes its 1-5 rubric that way, inside the scored input); empty otherwise.
static string AIScoreLevelDescription(const string &text, int64_t n) {
	const string num = std::to_string(n);
	size_t pos = 0;
	while (pos < text.size()) {
		size_t end = text.find('\n', pos);
		if (end == string::npos) {
			end = text.size();
		}
		size_t p = pos;
		while (p < end && (text[p] == ' ' || text[p] == '\t')) {
			p++;
		}
		if (text.compare(p, num.size(), num) == 0) {
			size_t q = p + num.size();
			const bool sep = q < end && (text[q] == ':' || text[q] == '.' || text[q] == ')' ||
			                             (text[q] == ' ' && q + 1 < end && text[q + 1] == '-'));
			if (sep) {
				q += text[q] == ' ' ? 2 : 1;
				while (q < end && text[q] == ' ') {
					q++;
				}
				if (q < end) {
					return text.substr(q, end - q);
				}
			}
		}
		pos = end + 1;
	}
	return string();
}
//! ai_score(input, criteria, lo, hi) with INTEGER bounds spanning 2..10 values -> a Score over the levels
//! lo..hi, low to high, with the scored input as the state and the criteria as the instructions. A level's
//! description is the rubric's own line for it when the criteria or the input list one, else its rank.
//! Other ai_score forms (no bounds, double bounds, more than 10 levels) keep the chat model.
static void AIMarkScoreQuestion(AIRequest &req, const string &input, const string &criteria, double lo, double hi,
                                bool is_int) {
	if (!is_int) {
		return;
	}
	const int64_t l = std::llround(lo), h = std::llround(hi);
	if (h < l + 1 || h - l + 1 > 10) {
		return;
	}
	req.question = AIRequest::Question::SCORE;
	req.state = input;
	req.score_lo = static_cast<double>(l);
	req.instructions = criteria.empty() ? string("Score the state on the listed scale.")
	                                    : "Score the state on the listed scale: " + criteria;
	for (int64_t n = l; n <= h; n++) {
		string desc = AIScoreLevelDescription(criteria, n);
		if (desc.empty()) {
			desc = AIScoreLevelDescription(input, n);
		}
		if (desc.empty()) {
			desc = "Score " + std::to_string(n) + " on a scale from " + std::to_string(l) + " (lowest) to " +
			       std::to_string(h) + " (highest).";
		}
		req.options.emplace_back(std::to_string(n), desc);
	}
}
static const char *const AI_COMPLETE_SYSTEM_PROMPT = "The user will provide an instruction and some relevant context.\n"
                                                     "Your job is to answer the user's instruction given the context.";
static const char *const AI_CLASSIFY_SYSTEM_PROMPT =
    "The user will provide a list of categories and some relevant context.\n"
    "Your job is to select the single listed category that best matches the given context.";
static const char *const AI_SCORE_SYSTEM_PROMPT =
    "The user will provide a scoring instruction and some relevant context.\n"
    "Your job is to score the given context per the instruction, returning a numeric score (higher is "
    "better) on a 0 to 1 scale, unless the request specifies a different scale.";
static const char *const AI_AGG_LEAF_SYSTEM_PROMPT =
    "Your job is to provide an answer to the user's instruction given the context below from multiple "
    "documents.\n"
    "Remember that your job is to answer the user's instruction by combining all relevant information from "
    "all provided documents, into a single coherent answer.\n"
    "Do NOT copy the format of the sources! Instead output your answer in a coherent, well-structured manner "
    "that best answers the user instruction.\n"
    "You have limited space to provide your answer, so be concise and to the point.";
static const char *const AI_AGG_NODE_SYSTEM_PROMPT =
    "Your job is to provide an answer to the user's instruction given the context below from multiple "
    "sources.\n"
    "Note that each source may be formatted differently and contain information about several different "
    "documents.\n"
    "Remember that your job is to answer the user's instruction by combining all relevant information from "
    "all provided sources, into a single coherent answer.\n"
    "The sources may provide opposing viewpoints or complementary information.\n"
    "Be sure to include information from ALL relevant sources in your answer.\n"
    "Do NOT copy the format of the sources, instead output your answer in a coherent, well-structured manner "
    "that best answers the user instruction.";

//===--------------------------------------------------------------------===//
// Implicit history keys: at bind time, collect the column references inside the
// prompt expression and append them as hidden arguments; their per-row values
// become the history keys. History is only active when ai_history_mode(true).
//===--------------------------------------------------------------------===//
struct AIKeyBindData : public FunctionData {
	AIKeyBindData(idx_t key_start, idx_t key_count, string pred_text = string())
	    : key_start(key_start), key_count(key_count), pred_text(std::move(pred_text)) {
	}
	idx_t key_start; // first appended key column (== the original user arg count)
	idx_t key_count; // number of appended key columns
	//! ai_filter only: the prompt's constant (question) part, bound-time evaluated -- the same
	//! AISplitPrompt split the reorder node carries per leaf, so the scalar path asks the TypeSafe
	//! backend the byte-identical question. Empty when the prompt has no constant/varying split.
	string pred_text;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<AIKeyBindData>(key_start, key_count, pred_text);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<AIKeyBindData>();
		return key_start == other.key_start && key_count == other.key_count && pred_text == other.pred_text;
	}
};

//! Bound-time predicate text of an ai_filter prompt expression (see AIKeyBindData::pred_text).
static string AIBindPredText(ClientContext &context, const Expression &prompt) {
	auto split = AISplitPrompt(context, prompt);
	if (!split.first || !split.first->IsFoldable() || split.first->Equals(*split.second)) {
		return string(); // no clean split: nothing to quote
	}
	try {
		Value v = ExpressionExecutor::EvaluateScalar(context, *split.first);
		return v.IsNull() ? string() : StringValue::Get(v);
	} catch (...) {
		return string();
	}
}

static void CollectColumnRefs(Expression &expr, vector<unique_ptr<Expression>> &out) {
	if (expr.GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
		// DEDUPED: the NULL guard below mentions each column twice (once in its IS NULL test, once
		// in the prompt), and a re-bind walks the guarded expression. Without dedup the key count
		// would grow on the second bind, which is the non-idempotence this function exists to
		// avoid.
		for (auto &seen : out) {
			if (seen->Equals(expr)) {
				return;
			}
		}
		out.push_back(expr.Copy());
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { CollectColumnRefs(child, out); });
}

//! Is this expression a string concatenation (the `||` operator binds to `concat`)?
static bool IsConcatCall(const Expression &expr) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	const auto &name = expr.Cast<BoundFunctionExpression>().Function().GetName();
	return name == "concat" || name == "||" || name == "concat_ws";
}

//! Replace each column reference that feeds a concatenation with COALESCE(col, ''), so ONE missing
//! column no longer erases the whole prompt. A prompt built by `||` is NULL if any operand is, and
//! the predicate then sees nothing at all -- not even the columns that DID have values.
//!
//! Only column refs sitting directly under a concat are touched, and only when EVERY column ref in
//! the prompt is in that position; anything used arithmetically (`'x' || (a + 1)`) is left alone,
//! since coalescing it to a string would change what is computed. Non-VARCHAR columns are cast
//! first, which is what the concatenation would have done anyway.
//!
//! Already-coalesced refs are skipped, so this is idempotent under re-binding like the rest of
//! AIKeyBind.
static bool CoalesceConcatColumns(ClientContext &context, unique_ptr<Expression> &expr, bool under_concat,
                                  bool &all_in_concat) {
	if (expr->GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
		if (!under_concat) {
			all_in_concat = false;
			return false;
		}
		auto column = std::move(expr);
		auto empty = make_uniq<BoundConstantExpression>(Value(""));
		vector<unique_ptr<Expression>> children;
		if (column->GetReturnType() != LogicalType::VARCHAR) {
			children.push_back(BoundCastExpression::AddCastToType(context, std::move(column), LogicalType::VARCHAR));
		} else {
			children.push_back(std::move(column));
		}
		children.push_back(std::move(empty));
		auto coalesce = make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_COALESCE, LogicalType::VARCHAR);
		coalesce->GetChildrenMutable() = std::move(children);
		expr = std::move(coalesce);
		return true;
	}
	if (expr->GetExpressionType() == ExpressionType::OPERATOR_COALESCE) {
		return false; // already guarded by an earlier bind
	}
	const bool concat_here = IsConcatCall(*expr);
	bool changed = false;
	ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
		if (CoalesceConcatColumns(context, child, concat_here, all_in_concat)) {
			changed = true;
		}
	});
	return changed;
}

//! True when the tail of `arguments` already carries `key_exprs`: a RE-BIND. DuckDB re-binds a function whenever
//! it duplicates a plan (LogicalOperator::Copy serializes and deserializes, CTE inlining copies the body per
//! reference), and by then the argument list already ends with the key columns the first bind appended.
//! Appending them again shifted key_start, grew the list (which could select another overload: ai_complete(prompt)
//! -> ai_complete(prompt, json_schema) read a key column as the schema and agent_bench Q26 returned 0 rows), and
//! since DuckDB 2.0 is refused outright ("cannot add or remove arguments in its bind callback"; SWAN 2.0
//! formula_1-23 and european_football_2-09, whose CTE is read twice). `min_user_args` = the arguments the
//! function takes before the keys (1: the prompt), so a user argument is never mistaken for a key.
static bool AIKeysAlreadyAppended(const vector<unique_ptr<Expression>> &arguments,
                                  const vector<unique_ptr<Expression>> &key_exprs, idx_t min_user_args) {
	const idx_t n = key_exprs.size();
	if (n == 0 || arguments.size() < min_user_args + n) {
		return false;
	}
	const idx_t start = arguments.size() - n;
	for (idx_t i = 0; i < n; i++) {
		if (!arguments[start + i] || !arguments[start + i]->Equals(*key_exprs[i])) {
			return false;
		}
	}
	return true;
}

static unique_ptr<FunctionData> AIKeyBind(BindScalarFunctionInput &input) {
	auto &arguments = input.GetArguments();
	auto &bound_function = input.GetBoundFunction();
	vector<unique_ptr<Expression>> key_exprs;
	if (!arguments.empty() && arguments[0]) {
		CollectColumnRefs(*arguments[0], key_exprs);
	}
	// IDEMPOTENCE: a re-bind returns the keys already in place (see AIKeysAlreadyAppended).
	const idx_t n = key_exprs.size();
	if (AIKeysAlreadyAppended(arguments, key_exprs, 1)) {
		return make_uniq<AIKeyBindData>(arguments.size() - n, n,
		                                bound_function.GetName() == "ai_filter"
		                                    ? AIBindPredText(input.GetClientContext(), *arguments[0])
		                                    : string());
	}
	// ai_filter takes exactly one argument, the prompt. Its varargs signature exists only so this bind can
	// append the key columns; a re-bind (keys already in place) returned above, so any other extra argument
	// was written by the user and would otherwise be silently ignored.
	if (bound_function.GetName() == "ai_filter" && arguments.size() != 1) {
		throw BinderException("ai_filter takes exactly one argument, the prompt (got %llu)",
		                      static_cast<uint64_t>(arguments.size()));
	}
	// NULL handling for the prompt, expressed IN THE EXPRESSION so every evaluation path inherits
	// it -- the scalar functions, and equally the folded reorder node, which rebuilds its prompts
	// from this expression and would otherwise need its own copy of the rule.
	//
	//     CASE WHEN c1 IS NULL AND ... AND cn IS NULL THEN NULL
	//          ELSE <prompt with each column coalesced to ''> END
	//
	// One missing column no longer erases the prompt: `'name: ' || a || ' desc: ' || b` is NULL in
	// SQL if EITHER is NULL, so the predicate never saw the column that did have a value. Now it
	// judges what is there. A row with nothing at all to judge still yields NULL -- unknown is the
	// honest answer, and no call is made for it.
	if (!key_exprs.empty() && arguments[0]) {
		bool all_in_concat = true;
		auto guarded = arguments[0]->Copy();
		const bool rewrote = CoalesceConcatColumns(input.GetClientContext(), guarded, false, all_in_concat);
		// `rewrote == false` means every column ref is already coalesced: this expression was
		// guarded by an earlier bind, and wrapping it again would nest the CASE on every re-bind.
		if (rewrote && all_in_concat) {
			unique_ptr<Expression> all_null;
			for (auto &key_expr : key_exprs) {
				auto is_null =
				    make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NULL, LogicalType::BOOLEAN);
				is_null->GetChildrenMutable().push_back(key_expr->Copy());
				if (!all_null) {
					all_null = std::move(is_null);
				} else {
					auto conj = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
					conj->GetChildrenMutable().push_back(std::move(all_null));
					conj->GetChildrenMutable().push_back(std::move(is_null));
					all_null = std::move(conj);
				}
			}
			auto null_value = make_uniq<BoundConstantExpression>(Value(LogicalType::VARCHAR));
			arguments[0] =
			    make_uniq<BoundCaseExpression>(std::move(all_null), std::move(null_value), std::move(guarded));
		}
	}
	const idx_t key_start = arguments.size();
	const string pred_text = bound_function.GetName() == "ai_filter" && !arguments.empty() && arguments[0]
	                             ? AIBindPredText(input.GetClientContext(), *arguments[0])
	                             : string();
	for (auto &key_expr : key_exprs) {
		bound_function.GetArguments().push_back(key_expr->GetReturnType());
		arguments.push_back(std::move(key_expr));
	}
	return make_uniq<AIKeyBindData>(key_start, key_exprs.size(), pred_text);
}

// Read the appended key-column values for one row into `keys`.
static void GetRowKeys(DataChunk &args, const AIKeyBindData &bind_data, idx_t row, vector<string> &keys) {
	keys.clear();
	for (idx_t j = 0; j < bind_data.key_count; j++) {
		Value value = args.data[bind_data.key_start + j].GetValue(row);
		if (!value.IsNull()) {
			keys.push_back(value.ToString());
		}
	}
}

//===--------------------------------------------------------------------===//
// AI_FILTER(prompt) -> BOOLEAN   (history keys implicit, from prompt columns)
//===--------------------------------------------------------------------===//
static void AIFilterFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().BindInfo()->Cast<AIKeyBindData>();
	const bool history_on = AIHistoryEnabled() && bind_data.key_count > 0;

	UnifiedVectorFormat prompt_format;
	args.data[0].ToUnifiedFormat(prompt_format);
	auto prompts = UnifiedVectorFormat::GetData<string_t>(prompt_format);

	vector<AIRequest> requests(count);
	vector<bool> valid(count, false);
	vector<vector<string>> row_keys(count);
	for (idx_t i = 0; i < count; i++) {
		auto idx = prompt_format.sel->get_index(i);
		if (!prompt_format.validity.RowIsValid(idx)) {
			continue;
		}
		valid[i] = true;
		requests[i].prompt = prompts[idx].GetString();
		requests[i].system_prompt = AIFilterSystemPrompt();
		requests[i].json_schema = AIFilterSchema();
		requests[i].schema_name = "ai_filter";
		AIMarkFilterQuestion(requests[i], bind_data.pred_text);
		if (history_on) {
			GetRowKeys(args, bind_data, i, row_keys[i]);
			requests[i].history = AIHistoryGather(row_keys[i]);
		}
	}

	auto responses = AIBatchCompleteFor(state.GetContext(), requests);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<bool>(result);
	auto &out_validity = FlatVector::ValidityMutable(result);
	for (idx_t i = 0; i < count; i++) {
		bool value;
		if (valid[i] && responses[i].success && AIReadFilterVerdict(responses[i].content, value)) {
			out[i] = value;
			if (history_on) {
				AIHistoryRecord(row_keys[i], requests[i].prompt, responses[i].content);
			}
		} else {
			out_validity.SetInvalid(i);
		}
	}
}

//===--------------------------------------------------------------------===//
// AI_CLASSIFY(input, categories[, instruction]) -> VARCHAR
//===--------------------------------------------------------------------===//
static void AIClassifyFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().BindInfo()->Cast<AIKeyBindData>();
	const bool history_on = AIHistoryEnabled() && bind_data.key_count > 0;
	const bool has_instruction = bind_data.key_start >= 3;

	UnifiedVectorFormat input_format;
	args.data[0].ToUnifiedFormat(input_format);
	auto inputs = UnifiedVectorFormat::GetData<string_t>(input_format);

	auto &cats_vec = args.data[1];
	UnifiedVectorFormat cats_format;
	cats_vec.ToUnifiedFormat(cats_format);
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(cats_format);
	auto &cats_child = ListVector::GetChild(cats_vec);
	UnifiedVectorFormat cats_child_format;
	cats_child.ToUnifiedFormat(cats_child_format);
	auto cats_child_data = UnifiedVectorFormat::GetData<string_t>(cats_child_format);

	UnifiedVectorFormat instruction_format;
	const string_t *instructions = nullptr;
	if (has_instruction) {
		args.data[2].ToUnifiedFormat(instruction_format);
		instructions = UnifiedVectorFormat::GetData<string_t>(instruction_format);
	}

	vector<AIRequest> requests(count);
	vector<bool> valid(count, false);
	vector<vector<string>> row_keys(count);
	for (idx_t i = 0; i < count; i++) {
		auto input_idx = input_format.sel->get_index(i);
		auto cats_idx = cats_format.sel->get_index(i);
		if (!input_format.validity.RowIsValid(input_idx) || !cats_format.validity.RowIsValid(cats_idx)) {
			continue;
		}
		auto entry = list_entries[cats_idx];
		string category_list;
		bool have_category = false;
		for (idx_t k = 0; k < entry.length; k++) {
			auto child_idx = cats_child_format.sel->get_index(entry.offset + k);
			if (!cats_child_format.validity.RowIsValid(child_idx)) {
				continue;
			}
			if (have_category) {
				category_list += ", ";
			}
			category_list += cats_child_data[child_idx].GetString();
			have_category = true;
		}
		if (!have_category) {
			continue;
		}

		string system = AI_CLASSIFY_SYSTEM_PROMPT;
		string extra_instruction;
		if (has_instruction) {
			auto instruction_idx = instruction_format.sel->get_index(i);
			if (instruction_format.validity.RowIsValid(instruction_idx)) {
				extra_instruction = instructions[instruction_idx].GetString();
				system += " " + extra_instruction;
			}
		}

		valid[i] = true;
		// Fixed schema -> the allowed categories go in the prompt.
		requests[i].prompt = "Categories: " + category_list + "\n\nInput:\n" + inputs[input_idx].GetString() +
		                     "\n\nRespond with exactly one of the listed categories, verbatim.";
		requests[i].system_prompt = std::move(system);
		requests[i].json_schema = STRING_RESULT_SCHEMA;
		requests[i].schema_name = "ai_classify";
		vector<std::pair<string, string>> options;
		for (idx_t k = 0; k < entry.length; k++) {
			auto child_idx = cats_child_format.sel->get_index(entry.offset + k);
			if (cats_child_format.validity.RowIsValid(child_idx)) {
				options.emplace_back(cats_child_data[child_idx].GetString(), string());
			}
		}
		AIMarkClassifyQuestion(requests[i], inputs[input_idx].GetString(), std::move(options), extra_instruction);
		if (history_on) {
			GetRowKeys(args, bind_data, i, row_keys[i]);
			requests[i].history = AIHistoryGather(row_keys[i]);
		}
	}

	auto responses = AIBatchCompleteFor(state.GetContext(), requests);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<string_t>(result);
	auto &out_validity = FlatVector::ValidityMutable(result);
	for (idx_t i = 0; i < count; i++) {
		string label;
		if (valid[i] && responses[i].success && AIParseStringField(responses[i].content, "result", label)) {
			out[i] = StringVector::AddString(result, label);
			if (history_on) {
				AIHistoryRecord(row_keys[i], requests[i].prompt, responses[i].content);
			}
		} else {
			out_validity.SetInvalid(i);
		}
	}
}

//===--------------------------------------------------------------------===//
// AI_CLASSIFY(input, categories_struct) -> VARCHAR
//
// categories is a STRUCT of {label: description}. The field names are the
// selectable labels (used to build an enum-constrained schema, fixed per query)
// and the field values describe each label to disambiguate the choice.
//===--------------------------------------------------------------------===//
static void AIClassifyStructFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().BindInfo()->Cast<AIKeyBindData>();
	const bool history_on = AIHistoryEnabled() && bind_data.key_count > 0;
	auto &cats_vec = args.data[1];
	if (cats_vec.GetType().id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("ai_classify: the categories argument must be a STRUCT of {label: description}");
	}
	auto &child_types = StructType::GetChildTypes(cats_vec.GetType());
	const idx_t n_fields = child_types.size();

	// Category labels are the struct field names (known statically).
	vector<string> labels;
	labels.reserve(n_fields);
	string enum_json = "[";
	for (idx_t f = 0; f < n_fields; f++) {
		if (f) {
			enum_json += ",";
		}
		const string &label = child_types[f].first.GetIdentifierName();
		labels.push_back(label);
		enum_json += "\"" + AIJsonEscape(label) + "\"";
	}
	enum_json += "]";
	// Enum-constrained schema (fixed per query since the struct type is fixed).
	const string schema = "{\"type\":\"object\",\"properties\":{\"result\":{\"type\":\"string\",\"enum\":" + enum_json +
	                      "}},\"required\":[\"result\"],\"additionalProperties\":false}";

	UnifiedVectorFormat input_format;
	args.data[0].ToUnifiedFormat(input_format);
	auto inputs = UnifiedVectorFormat::GetData<string_t>(input_format);

	vector<AIRequest> requests(count);
	vector<bool> valid(count, false);
	vector<vector<string>> row_keys(count);
	for (idx_t i = 0; i < count; i++) {
		auto input_idx = input_format.sel->get_index(i);
		if (!input_format.validity.RowIsValid(input_idx)) {
			continue;
		}
		Value struct_val = cats_vec.GetValue(i);
		if (struct_val.IsNull()) {
			continue;
		}
		auto &descriptions = StructValue::GetChildren(struct_val);
		string category_block;
		vector<std::pair<string, string>> options;
		for (idx_t f = 0; f < n_fields; f++) {
			category_block += "- " + labels[f];
			string desc;
			if (f < descriptions.size() && !descriptions[f].IsNull()) {
				desc = descriptions[f].ToString();
				if (!desc.empty() && desc != labels[f]) {
					category_block += ": " + desc;
				}
			}
			category_block += "\n";
			options.emplace_back(labels[f], desc == labels[f] ? string() : desc);
		}
		valid[i] = true;
		requests[i].system_prompt = AI_CLASSIFY_SYSTEM_PROMPT;
		requests[i].prompt = "Categories (name: description):\n" + category_block + "\nInput:\n" +
		                     inputs[input_idx].GetString() + "\n\nSelect exactly one category by its name.";
		requests[i].json_schema = schema;
		requests[i].schema_name = "ai_classify";
		AIMarkClassifyQuestion(requests[i], inputs[input_idx].GetString(), std::move(options), string());
		if (history_on) {
			GetRowKeys(args, bind_data, i, row_keys[i]);
			requests[i].history = AIHistoryGather(row_keys[i]);
		}
	}

	auto responses = AIBatchCompleteFor(state.GetContext(), requests);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<string_t>(result);
	auto &out_validity = FlatVector::ValidityMutable(result);
	for (idx_t i = 0; i < count; i++) {
		string label;
		if (valid[i] && responses[i].success && AIParseStringField(responses[i].content, "result", label)) {
			out[i] = StringVector::AddString(result, label);
			if (history_on) {
				AIHistoryRecord(row_keys[i], requests[i].prompt, responses[i].content);
			}
		} else {
			out_validity.SetInvalid(i);
		}
	}
}

//===--------------------------------------------------------------------===//
// AI_SCORE(input[, criteria[, lo, hi]]) -> DOUBLE (default 0-1) or BIGINT (integer bounds)
// History keys are implicit: the columns referenced in `input` (see AIKeyBind).
// A user range: ai_score(input, criteria, lo, hi). Integer bounds -> BIGINT in [lo,hi] (rounded); float
// bounds -> DOUBLE in [lo,hi]. The range goes into the USER prompt so the reorder can bake it identically.
//===--------------------------------------------------------------------===//

// Format a score bound for the prompt: integers without decimals, doubles with trailing zeros stripped.
static string AIFormatBound(double v, bool is_int) {
	if (is_int) {
		return std::to_string(static_cast<int64_t>(std::llround(v)));
	}
	string s = std::to_string(v);
	if (s.find('.') != string::npos) {
		s.erase(s.find_last_not_of('0') + 1);
		if (!s.empty() && s.back() == '.') {
			s.pop_back();
		}
	}
	return s;
}

// Build ai_score's user prompt. Kept BYTE-IDENTICAL with the reorder's baked prompt
// (ai_filter_tree_build.cpp); the result-preservation test guards any drift.
static string AIScoreUserPrompt(bool has_criteria, const string &criteria, const string &input, bool has_range,
                                double lo, double hi, bool is_int) {
	if (has_range) {
		return "Provide a score from " + AIFormatBound(lo, is_int) + " to " + AIFormatBound(hi, is_int) +
		       (is_int ? " as an integer." : ".") + "\n\nScoring criteria: " + criteria + "\n\nInput:\n" + input;
	}
	if (has_criteria) {
		return "Scoring criteria: " + criteria + "\n\nInput:\n" + input;
	}
	return input;
}

// Bind data for ai_score: AIKeyBind's implicit history keys (inherited) plus an optional user range/type.
// Inherits AIKeyBindData so GetRowKeys() works unchanged.
struct AIScoreBindData : public AIKeyBindData {
	AIScoreBindData(idx_t key_start, idx_t key_count, bool has_range, double lo, double hi, bool is_int)
	    : AIKeyBindData(key_start, key_count), has_range(has_range), lo(lo), hi(hi), is_int(is_int) {
	}
	bool has_range; // ai_score(input, criteria, lo, hi) was used
	double lo;
	double hi;
	bool is_int; // integer bounds -> BIGINT output, rounded
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<AIScoreBindData>(key_start, key_count, has_range, lo, hi, is_int);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &o = other_p.Cast<AIScoreBindData>();
		return key_start == o.key_start && key_count == o.key_count && has_range == o.has_range && lo == o.lo &&
		       hi == o.hi && is_int == o.is_int;
	}
};

static unique_ptr<FunctionData> AIScoreBind(BindScalarFunctionInput &input) {
	auto &arguments = input.GetArguments();
	auto &bound_function = input.GetBoundFunction();
	// Implicit history keys from the input (arg 0), exactly like AIKeyBind; on a re-bind they are already in
	// place and the argument count must not change (AIKeysAlreadyAppended).
	vector<unique_ptr<Expression>> key_exprs;
	if (!arguments.empty() && arguments[0]) {
		CollectColumnRefs(*arguments[0], key_exprs);
	}
	const bool rebind = AIKeysAlreadyAppended(arguments, key_exprs, 1);
	// ai_score(input, criteria, lo, hi): read the constant bounds; int-ness comes from the overload (the
	// BIGINT-bounds overload -> integer output, rounded; the DOUBLE-bounds overload -> double output).
	const idx_t declared = rebind ? arguments.size() - key_exprs.size() : arguments.size();
	bool has_range = false;
	double lo = 0, hi = 1;
	bool is_int = false;
	if (declared >= 4 && arguments[2] && arguments[3]) {
		is_int = arguments[2]->GetReturnType().IsIntegral();
		try {
			lo = ExpressionExecutor::EvaluateScalar(input.GetClientContext(), *arguments[2]).GetValue<double>();
			hi = ExpressionExecutor::EvaluateScalar(input.GetClientContext(), *arguments[3]).GetValue<double>();
			has_range = true;
		} catch (...) {
			has_range = false; // non-constant bounds -> fall back to the default 0-1 double scale
		}
	}
	if (rebind) {
		return make_uniq<AIScoreBindData>(declared, key_exprs.size(), has_range, lo, hi, is_int);
	}
	const idx_t key_start = arguments.size();
	for (auto &key_expr : key_exprs) {
		bound_function.GetArguments().push_back(key_expr->GetReturnType());
		arguments.push_back(std::move(key_expr));
	}
	return make_uniq<AIScoreBindData>(key_start, key_exprs.size(), has_range, lo, hi, is_int);
}

static void AIScoreFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &bind_data = func_expr.BindInfo()->Cast<AIScoreBindData>();
	const bool history_on = AIHistoryEnabled() && bind_data.key_count > 0;
	const bool has_criteria = bind_data.key_start >= 2;
	const bool has_range = bind_data.has_range;

	UnifiedVectorFormat input_format;
	args.data[0].ToUnifiedFormat(input_format);
	auto inputs = UnifiedVectorFormat::GetData<string_t>(input_format);

	UnifiedVectorFormat criteria_format;
	const string_t *criteria = nullptr;
	if (has_criteria) {
		args.data[1].ToUnifiedFormat(criteria_format);
		criteria = UnifiedVectorFormat::GetData<string_t>(criteria_format);
	}

	vector<AIRequest> requests(count);
	vector<bool> valid(count, false);
	vector<vector<string>> row_keys(count);
	for (idx_t i = 0; i < count; i++) {
		auto input_idx = input_format.sel->get_index(i);
		if (!input_format.validity.RowIsValid(input_idx)) {
			continue;
		}
		string criteria_text;
		if (has_criteria) {
			auto criteria_idx = criteria_format.sel->get_index(i);
			criteria_text =
			    criteria_format.validity.RowIsValid(criteria_idx) ? criteria[criteria_idx].GetString() : string();
		}
		valid[i] = true;
		requests[i].prompt = AIScoreUserPrompt(has_criteria, criteria_text, inputs[input_idx].GetString(), has_range,
		                                       bind_data.lo, bind_data.hi, bind_data.is_int);
		requests[i].system_prompt = AI_SCORE_SYSTEM_PROMPT;
		requests[i].json_schema = NUMBER_RESULT_SCHEMA;
		requests[i].schema_name = "ai_score";
		if (has_range) {
			AIMarkScoreQuestion(requests[i], inputs[input_idx].GetString(), criteria_text, bind_data.lo, bind_data.hi,
			                    bind_data.is_int);
		}
		if (history_on) {
			GetRowKeys(args, bind_data, i, row_keys[i]);
			requests[i].history = AIHistoryGather(row_keys[i]);
		}
	}

	auto responses = AIBatchCompleteFor(state.GetContext(), requests);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	const bool int_out = has_range && bind_data.is_int; // integer bounds -> BIGINT result vector
	auto out_dbl = int_out ? nullptr : FlatVector::GetDataMutable<double>(result);
	auto out_int = int_out ? FlatVector::GetDataMutable<int64_t>(result) : nullptr;
	auto &out_validity = FlatVector::ValidityMutable(result);
	for (idx_t i = 0; i < count; i++) {
		double value;
		if (valid[i] && responses[i].success && AIParseDoubleField(responses[i].content, "result", value)) {
			if (has_range) { // clamp to [lo, hi]
				value = value < bind_data.lo ? bind_data.lo : (value > bind_data.hi ? bind_data.hi : value);
			}
			if (int_out) {
				out_int[i] = static_cast<int64_t>(std::llround(value));
			} else {
				out_dbl[i] = value;
			}
			if (history_on) {
				AIHistoryRecord(row_keys[i], requests[i].prompt, responses[i].content);
			}
		} else {
			out_validity.SetInvalid(i);
		}
	}
}

//===--------------------------------------------------------------------===//
// AI_AGG(items LIST(VARCHAR), task) -> VARCHAR   (use with list()/GROUP BY)
//
// Hierarchical map-reduce: items too large for one context are split into
// budget-sized groups (map -> partial results), which are then combined
// (reduce), recursing until a single result remains. Batched across rows.
//===--------------------------------------------------------------------===//
static vector<vector<string>> PartitionByBudget(const vector<string> &items, idx_t char_budget) {
	vector<vector<string>> groups;
	vector<string> current;
	idx_t current_size = 0;
	for (const auto &item : items) {
		const idx_t item_size = item.size() + 3; // "- " + "\n"
		if (!current.empty() && current_size + item_size > char_budget) {
			groups.push_back(std::move(current));
			current.clear();
			current_size = 0;
		}
		current.push_back(item);
		current_size += item_size;
	}
	if (!current.empty()) {
		groups.push_back(std::move(current));
	}
	return groups;
}

static AIRequest BuildAggRequest(const string &task, const vector<string> &items, bool combine_partials) {
	string body;
	for (const auto &item : items) {
		body += "- " + item + "\n";
	}
	AIRequest request;
	request.schema_name = "ai_agg";
	request.json_schema = STRING_RESULT_SCHEMA;
	if (combine_partials) {
		request.system_prompt = AI_AGG_NODE_SYSTEM_PROMPT;
		request.prompt = "Instruction: " + task + "\n\nPartial results:\n" + body;
	} else {
		request.system_prompt = AI_AGG_LEAF_SYSTEM_PROMPT;
		request.prompt = "Instruction: " + task + "\n\nItems:\n" + body;
	}
	return request;
}

struct AggRowState {
	bool valid = false;
	bool done = false;
	bool partial = false; // are `parts` partial results (reduce) or raw items (map)?
	string task;
	vector<string> parts;
	AIResult result;
};

static void AIAggFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	const idx_t char_budget = AIConfig::Get().agg_char_budget;
	static constexpr idx_t MAX_ROUNDS = 24;

	auto &list_vec = args.data[0];
	UnifiedVectorFormat list_format;
	list_vec.ToUnifiedFormat(list_format);
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_format);
	auto &list_child = ListVector::GetChild(list_vec);
	UnifiedVectorFormat list_child_format;
	list_child.ToUnifiedFormat(list_child_format);
	auto list_child_data = UnifiedVectorFormat::GetData<string_t>(list_child_format);

	UnifiedVectorFormat task_format;
	args.data[1].ToUnifiedFormat(task_format);
	auto tasks = UnifiedVectorFormat::GetData<string_t>(task_format);

	// Seed per-row state with the raw items.
	vector<AggRowState> states(count);
	for (idx_t i = 0; i < count; i++) {
		auto list_idx = list_format.sel->get_index(i);
		auto task_idx = task_format.sel->get_index(i);
		if (!list_format.validity.RowIsValid(list_idx) || !task_format.validity.RowIsValid(task_idx)) {
			states[i].done = true;
			continue;
		}
		auto entry = list_entries[list_idx];
		for (idx_t k = 0; k < entry.length; k++) {
			auto child_idx = list_child_format.sel->get_index(entry.offset + k);
			if (list_child_format.validity.RowIsValid(child_idx)) {
				states[i].parts.push_back(list_child_data[child_idx].GetString());
			}
		}
		if (states[i].parts.empty()) {
			states[i].done = true;
			continue;
		}
		states[i].valid = true;
		states[i].task = tasks[task_idx].GetString();
	}

	// Round-based map-reduce, batched across all active rows for concurrency.
	for (idx_t round = 0; round < MAX_ROUNDS; round++) {
		vector<AIRequest> requests;
		vector<idx_t> row_req_start(count, 0);
		vector<idx_t> row_req_count(count, 0);
		vector<bool> row_final(count, false);
		bool any_active = false;
		for (idx_t i = 0; i < count; i++) {
			if (!states[i].valid || states[i].done) {
				continue;
			}
			any_active = true;
			auto groups = PartitionByBudget(states[i].parts, char_budget);
			row_req_start[i] = requests.size();
			row_req_count[i] = groups.size();
			row_final[i] = (groups.size() == 1);
			for (auto &group : groups) {
				requests.push_back(BuildAggRequest(states[i].task, group, states[i].partial));
			}
		}
		if (!any_active) {
			break;
		}
		auto responses = AIBatchCompleteFor(state.GetContext(), requests);
		for (idx_t i = 0; i < count; i++) {
			if (!states[i].valid || states[i].done || row_req_count[i] == 0) {
				continue;
			}
			if (row_final[i]) {
				// A single group means this round produced the final consolidated result.
				auto &response = responses[row_req_start[i]];
				if (response.success) {
					string parsed;
					states[i].result.content =
					    AIParseStringField(response.content, "result", parsed) ? parsed : response.content;
					states[i].result.success = true;
				}
				states[i].done = true;
				continue;
			}
			// Map produced one partial per group; feed them into the next (reduce) round.
			vector<string> next;
			for (idx_t k = 0; k < row_req_count[i]; k++) {
				auto &response = responses[row_req_start[i] + k];
				if (response.success) {
					string parsed;
					next.push_back(AIParseStringField(response.content, "result", parsed) ? parsed : response.content);
				}
			}
			if (next.empty()) {
				states[i].done = true; // result stays a failure -> NULL
			} else {
				states[i].parts = std::move(next);
				states[i].partial = true;
			}
		}
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<string_t>(result);
	auto &out_validity = FlatVector::ValidityMutable(result);
	for (idx_t i = 0; i < count; i++) {
		if (states[i].valid && states[i].result.success) {
			out[i] = StringVector::AddString(result, states[i].result.content);
		} else {
			out_validity.SetInvalid(i);
		}
	}
}

//===--------------------------------------------------------------------===//
// AI_PROMPT(prompt) / AI_PROMPT(prompt, json_schema) -> VARCHAR
//===--------------------------------------------------------------------===//
static void AIPromptFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().BindInfo()->Cast<AIKeyBindData>();
	const bool history_on = AIHistoryEnabled() && bind_data.key_count > 0;
	const bool has_schema = bind_data.key_start >= 2; // 2nd user arg is the JSON schema

	UnifiedVectorFormat prompt_format;
	args.data[0].ToUnifiedFormat(prompt_format);
	auto prompts = UnifiedVectorFormat::GetData<string_t>(prompt_format);

	UnifiedVectorFormat schema_format;
	const string_t *schemas = nullptr;
	if (has_schema) {
		args.data[1].ToUnifiedFormat(schema_format);
		schemas = UnifiedVectorFormat::GetData<string_t>(schema_format);
	}

	vector<AIRequest> requests(count);
	vector<bool> valid(count, false);
	vector<vector<string>> row_keys(count);
	for (idx_t i = 0; i < count; i++) {
		auto prompt_idx = prompt_format.sel->get_index(i);
		if (!prompt_format.validity.RowIsValid(prompt_idx)) {
			continue;
		}
		valid[i] = true;
		requests[i].prompt = prompts[prompt_idx].GetString();
		requests[i].system_prompt = AI_COMPLETE_SYSTEM_PROMPT;
		requests[i].schema_name = "ai_complete";
		if (has_schema) {
			auto schema_idx = schema_format.sel->get_index(i);
			if (schema_format.validity.RowIsValid(schema_idx)) {
				requests[i].json_schema = schemas[schema_idx].GetString();
			}
		}
		if (history_on) {
			GetRowKeys(args, bind_data, i, row_keys[i]);
			requests[i].history = AIHistoryGather(row_keys[i]);
		}
	}

	auto responses = AIBatchCompleteFor(state.GetContext(), requests);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<string_t>(result);
	auto &out_validity = FlatVector::ValidityMutable(result);
	for (idx_t i = 0; i < count; i++) {
		if (valid[i] && responses[i].success) {
			out[i] = StringVector::AddString(result, responses[i].content);
			if (history_on) {
				AIHistoryRecord(row_keys[i], requests[i].prompt, responses[i].content);
			}
		} else {
			out_validity.SetInvalid(i);
		}
	}
}

//===--------------------------------------------------------------------===//
// AI_USAGE() -> one row per query (query_id, query_text, tokens, cost);
// AI_USAGE_RESET() -> BIGINT (number of query records cleared)
//===--------------------------------------------------------------------===//
struct AIUsageData : public GlobalTableFunctionState {
	AIUsageData() : offset(0) {
	}
	vector<AIQueryUsage> usage;
	idx_t offset;
};

static unique_ptr<FunctionData> AIUsageBind(ClientContext &context, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types, vector<Identifier> &names) {
	names = {"query_id",      "query_text",    "llm_calls",        "cache_hits",   "input_tokens",
	         "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cost_usd",
	         "embed_calls",   "embed_tokens",  "hedged_calls",     "failed_calls"};
	return_types = {LogicalType::UBIGINT, LogicalType::VARCHAR, LogicalType::UBIGINT, LogicalType::UBIGINT,
	                LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT,
	                LogicalType::UBIGINT, LogicalType::DOUBLE,  LogicalType::UBIGINT, LogicalType::UBIGINT,
	                LogicalType::UBIGINT, LogicalType::UBIGINT};
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> AIUsageInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<AIUsageData>();
	result->usage = AIGetUsage();
	return std::move(result);
}

static void AIUsageFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<AIUsageData>();
	const idx_t remaining = data.usage.size() - data.offset;
	const idx_t this_count = remaining < STANDARD_VECTOR_SIZE ? remaining : STANDARD_VECTOR_SIZE;
	output.SetChildCardinality(this_count);
	if (this_count == 0) {
		return;
	}
	for (idx_t row = 0; row < this_count; row++) {
		const auto &u = data.usage[data.offset + row];
		output.data[0].SetValue(row, Value::UBIGINT(u.query_id));
		output.data[1].SetValue(row, Value(u.query_text));
		output.data[2].SetValue(row, Value::UBIGINT(u.llm_calls));
		output.data[3].SetValue(row, Value::UBIGINT(u.cache_hits));
		output.data[4].SetValue(row, Value::UBIGINT(u.input_tokens));
		output.data[5].SetValue(row, Value::UBIGINT(u.cached_tokens));
		output.data[6].SetValue(row, Value::UBIGINT(u.output_tokens));
		output.data[7].SetValue(row, Value::UBIGINT(u.reasoning_tokens));
		output.data[8].SetValue(row, Value::UBIGINT(u.total_tokens));
		output.data[9].SetValue(row, Value::DOUBLE(u.cost_usd));
		output.data[10].SetValue(row, Value::UBIGINT(u.embed_calls));
		output.data[11].SetValue(row, Value::UBIGINT(u.embed_tokens));
		output.data[12].SetValue(row, Value::UBIGINT(u.hedged_calls));
		output.data[13].SetValue(row, Value::UBIGINT(u.failed_calls));
	}
	data.offset += this_count;
}

//===--------------------------------------------------------------------===//
// ai_filter_training_data() -> (prompt, embedding FLOAT[], label BOOLEAN)
// Self-labeled examples collected from real ai_filter calls; input for the selectivity MLP.
//===--------------------------------------------------------------------===//
struct AITrainingTableState : public GlobalTableFunctionState {
	AITrainingTableState() : offset(0) {
	}
	vector<AITrainingExample> data;
	idx_t offset;
};

static unique_ptr<FunctionData> AITrainingBind(ClientContext &context, TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types, vector<Identifier> &names) {
	names = {"prompt", "embedding", "label"};
	return_types = {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::BOOLEAN};
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> AITrainingInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_uniq<AITrainingTableState>();
	result->data = AIGetTrainingData();
	return std::move(result);
}

static void AITrainingFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<AITrainingTableState>();
	const idx_t remaining = data.data.size() - data.offset;
	const idx_t this_count = remaining < STANDARD_VECTOR_SIZE ? remaining : STANDARD_VECTOR_SIZE;
	output.SetChildCardinality(this_count);
	if (this_count == 0) {
		return;
	}
	for (idx_t row = 0; row < this_count; row++) {
		const auto &ex = data.data[data.offset + row];
		output.data[0].SetValue(row, Value(ex.prompt));
		vector<Value> floats;
		floats.reserve(ex.embedding.size());
		for (float f : ex.embedding) {
			floats.push_back(Value::FLOAT(f));
		}
		output.data[1].SetValue(row, Value::LIST(LogicalType::FLOAT, std::move(floats)));
		output.data[2].SetValue(row, Value::BOOLEAN(ex.label));
	}
	data.offset += this_count;
}

static void AITrainingResetFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const int64_t cleared = static_cast<int64_t>(AIResetTrainingData());
	const idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<int64_t>(result);
	for (idx_t i = 0; i < count; i++) {
		out[i] = cleared;
	}
}

static void AIUsageResetFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const int64_t prior_calls = static_cast<int64_t>(AIResetUsage());
	const idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<int64_t>(result);
	for (idx_t i = 0; i < count; i++) {
		out[i] = prior_calls;
	}
}

static void AIHistoryResetFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const int64_t cleared = static_cast<int64_t>(AIHistoryReset());
	const idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<int64_t>(result);
	for (idx_t i = 0; i < count; i++) {
		out[i] = cleared;
	}
}

// ai_history_mode(BOOLEAN) -> BOOLEAN : enable/disable implicit history; echoes the value.
static void AIHistoryModeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	UnifiedVectorFormat format;
	args.data[0].ToUnifiedFormat(format);
	auto enabled = UnifiedVectorFormat::GetData<bool>(format);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<bool>(result);
	auto &out_validity = FlatVector::ValidityMutable(result);
	for (idx_t i = 0; i < count; i++) {
		auto idx = format.sel->get_index(i);
		if (format.validity.RowIsValid(idx)) {
			AIHistorySetEnabled(enabled[idx]);
			out[i] = enabled[idx];
		} else {
			out_validity.SetInvalid(i);
		}
	}
}

// ai_turbo(BOOLEAN) -> BOOLEAN : enable/disable adaptive (AIMD) concurrency; echoes the value.
static void AITurboModeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	UnifiedVectorFormat format;
	args.data[0].ToUnifiedFormat(format);
	auto enabled = UnifiedVectorFormat::GetData<bool>(format);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<bool>(result);
	auto &out_validity = FlatVector::ValidityMutable(result);
	for (idx_t i = 0; i < count; i++) {
		auto idx = format.sel->get_index(i);
		if (format.validity.RowIsValid(idx)) {
			AITurboSetEnabled(enabled[idx]);
			out[i] = enabled[idx];
		} else {
			out_validity.SetInvalid(i);
		}
	}
}

// ai_image(VARCHAR path_or_url) -> VARCHAR : wrap an image reference in sentinels (\x01 .. \x02) so the AI
// functions send it to the multimodal model as an image, not as literal text. Use it inside a prompt via
// concatenation, e.g. ai_filter('Is this the logo of ' || track || '? ' || ai_image(image_path)). Local
// file paths are read + base64-encoded when the request is built; http(s) URLs are passed through.
static void AIImageFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	UnifiedVectorFormat format;
	args.data[0].ToUnifiedFormat(format);
	auto in = UnifiedVectorFormat::GetData<string_t>(format);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<string_t>(result);
	auto &out_validity = FlatVector::ValidityMutable(result);
	for (idx_t i = 0; i < count; i++) {
		auto idx = format.sel->get_index(i);
		if (!format.validity.RowIsValid(idx)) {
			out_validity.SetInvalid(i);
			continue;
		}
		string wrapped;
		wrapped.push_back(AI_IMAGE_OPEN);
		wrapped += in[idx].GetString();
		wrapped.push_back(AI_IMAGE_CLOSE);
		out[i] = StringVector::AddString(result, wrapped);
	}
}

//===--------------------------------------------------------------------===//
// SPECULATIVE_AI_FILTER(prompt) -> BOOLEAN
// Placeholder left at the pushed-down position after an ai_filter is pulled up above the joins.
// Its job (future) is to cheaply estimate, via embedding-based selectivity, whether an input row
// can be dropped before the joins -- reducing rows the real (pulled-up) ai_filter must verify.
// For now it passes every row (returns true), so query results are unchanged. It is VOLATILE so
// the optimizer never folds it away or removes the node.
//===--------------------------------------------------------------------===//
static void AISpeculativeFilterFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<bool>(result);
	for (idx_t i = 0; i < count; i++) {
		out[i] = true; // pass-through; selectivity-based row reduction to be added later
	}
}

//===--------------------------------------------------------------------===//
// AI_EMBED(text) -> FLOAT[]   (no history; fixed path; local sentence-transformers by default)
//===--------------------------------------------------------------------===//
static void AIEmbedFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	UnifiedVectorFormat fmt;
	args.data[0].ToUnifiedFormat(fmt);
	auto texts = UnifiedVectorFormat::GetData<string_t>(fmt);

	vector<string> inputs(count);
	vector<bool> valid(count, false);
	for (idx_t i = 0; i < count; i++) {
		auto idx = fmt.sel->get_index(i);
		if (fmt.validity.RowIsValid(idx)) {
			inputs[i] = texts[idx].GetString();
			valid[i] = true;
		}
	}

	auto embeddings = AIEmbedBatch(inputs, state.GetContext().GetCurrentQuery());

	result.SetVectorType(VectorType::FLAT_VECTOR);
	idx_t total = 0;
	for (idx_t i = 0; i < count; i++) {
		if (valid[i] && embeddings[i].success) {
			total += embeddings[i].embedding.size();
		}
	}
	ListVector::Reserve(result, total);
	auto list_data = FlatVector::GetDataMutable<list_entry_t>(result);
	auto child_data = FlatVector::GetDataMutable<float>(ListVector::GetChildMutable(result));
	auto &validity = FlatVector::ValidityMutable(result);
	idx_t offset = 0;
	for (idx_t i = 0; i < count; i++) {
		list_data[i].offset = offset;
		if (valid[i] && embeddings[i].success) {
			auto &e = embeddings[i].embedding;
			list_data[i].length = e.size();
			for (idx_t j = 0; j < e.size(); j++) {
				child_data[offset + j] = e[j];
			}
			offset += e.size();
		} else {
			list_data[i].length = 0;
			validity.SetInvalid(i);
		}
	}
	ListVector::SetListSize(result, offset);
}

//===--------------------------------------------------------------------===//
// AI_PREDICATE(tree_str, prompt_0..prompt_{n-1}, emb_0..emb_{n-1}) -> BOOLEAN
// Evaluates a boolean tree (AND/OR/NOT) over ai_filter leaves, choosing the evaluation order PER ROW
// from embedding-based selectivity estimates so the expensive LLM calls short-circuit early. Inserted
// by the DUCKDB_AI_REORDER optimizer rewrite; also directly callable (tree as a literal) for testing.
//===--------------------------------------------------------------------===//
//! Speculative gate: a row whose estimated P(pass) reaches this passes through to the pulled-up predicate
//! unevaluated; below it the leaf is evaluated (and pruned when false) before the join.
static constexpr double AI_SPECULATIVE_THRESHOLD = 0.5;

struct AIFilterWithEmbedBindData : public FunctionData {
	AIFilterWithEmbedBindData(shared_ptr<AIFilterTreeNode> tree, string tree_str, idx_t leaf_count,
	                          bool speculative = false, double threshold = 0.5, int64_t limit = -1, string meta = "",
	                          vector<shared_ptr<Expression>> wrappers = {})
	    : tree(std::move(tree)), tree_str(std::move(tree_str)), leaf_count(leaf_count), meta(std::move(meta)),
	      wrappers(std::move(wrappers)), speculative(speculative), threshold(threshold), limit(limit) {
	}
	shared_ptr<AIFilterTreeNode> tree;
	string tree_str;
	idx_t leaf_count;
	//! LIMIT stand-down: a LIMIT k was pushed into the evaluation ABOVE this speculative leaf, so
	//! eager leaf work (embeds + likely-fail evaluations) cannot pay off - pass every row through.
	bool stand_down = false;
	// Per-leaf kind + comparison for the generalized ai_predicate node, one leaf per ';': "F" (ai_filter,
	// boolean) or "<K>:<op>:<val>" where K in {C,S,M} (classify/score/complete), op in {eq,ne,gt,lt,ge,le,
	// in,ni,wrap}. Empty (ai_function_with_embed) means every leaf is an ai_filter. Parsed in AIFilterEvaluateBatch.
	string meta;
	//! Per leaf, the deserialized wrapper of a `wrap` token (null for the other leaves; empty when none):
	//! the boolean expression around the call, over a reference to column 0 = the answer. Immutable once
	//! bound, so copies of the bind data share them.
	vector<shared_ptr<Expression>> wrappers;
	// Speculative (row-adaptive partial pushdown) mode: only evaluate rows the MLP estimates are
	// likely to FAIL (roll-up P(pass) < threshold), pruning them before the join; pass the rest
	// through for the pulled-up filter to re-check. Off for the plain reorder node.
	bool speculative;
	double threshold;
	// LIMIT k pushed in from a LIMIT directly above this filter (AILimitPushdown): stop evaluating once k
	// rows have PASSED; the rest stay false, which is sound because the LIMIT keeps only k anyway. -1 = no
	// limit. Only set when the filter's boolean result feeds nothing but that LIMIT.
	int64_t limit;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<AIFilterWithEmbedBindData>(tree, tree_str, leaf_count, speculative, threshold, limit, meta,
		                                            wrappers);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<AIFilterWithEmbedBindData>();
		return tree_str == other.tree_str && leaf_count == other.leaf_count && speculative == other.speculative &&
		       threshold == other.threshold && limit == other.limit && meta == other.meta;
	}
};

// Split a meta_str into its per-leaf tokens (';'-separated; "" -> no tokens).
static vector<string> AIMetaTokens(const string &meta) {
	vector<string> tokens;
	if (meta.empty()) {
		return tokens;
	}
	string cur;
	for (const char c : meta) {
		if (c == ';') {
			tokens.push_back(cur);
			cur.clear();
		} else {
			cur += c;
		}
	}
	tokens.push_back(cur);
	return tokens;
}

// The base64 wrapper of a `<K>:wrap:<b64>[@range]` token, or "" for any other token.
static string AIMetaWrapperText(const string &tok) {
	static const string kWrap = ":wrap:";
	const auto at = tok.find(kWrap);
	if (at == string::npos) {
		return string();
	}
	const auto begin = at + kWrap.size();
	const auto end = tok.find('@', begin);
	return tok.substr(begin, end == string::npos ? string::npos : end - begin);
}

// Deserialize every wrapped leaf's wrapper (re-binding its functions in `context`); one slot per leaf.
static vector<shared_ptr<Expression>> AILeafWrappers(ClientContext &context, const string &meta, idx_t n) {
	vector<shared_ptr<Expression>> out;
	const auto tokens = AIMetaTokens(meta);
	for (idx_t l = 0; l < n && l < tokens.size(); l++) {
		const string b64 = AIMetaWrapperText(tokens[l]);
		if (b64.empty()) {
			continue;
		}
		out.resize(n);
		out[l] = AIDeserializeExpression(context, b64);
	}
	return out;
}

static unique_ptr<FunctionData> AIFilterWithEmbedBindImpl(BindScalarFunctionInput &input, bool speculative) {
	auto &arguments = input.GetArguments();
	if (arguments.empty() || !arguments[0] || arguments[0]->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
		throw BinderException("ai_function_with_embed: first argument must be a constant tree string");
	}
	auto &value = arguments[0]->Cast<BoundConstantExpression>().GetValue();
	if (value.IsNull()) {
		throw BinderException("ai_function_with_embed: tree string must not be NULL");
	}
	const string tree_str = StringValue::Get(value);
	auto parsed = AIFilterTreeParse(tree_str);
	if (!parsed) {
		throw BinderException("ai_function_with_embed: malformed tree '" + tree_str + "'");
	}
	const idx_t leaf_count = parsed->LeafCount();
	// Optional trailing meta_str: per-leaf kind + comparison for a generalized (mixed AI function) node.
	// Absent (arg count == 1 + 3n) means every leaf is a plain ai_filter -- unchanged behavior.
	const idx_t base_args = 1 + 3 * leaf_count;
	string meta;
	if (arguments.size() == base_args + 1) {
		auto &m = arguments[base_args];
		if (m && m->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
			auto &mv = m->Cast<BoundConstantExpression>().GetValue();
			if (!mv.IsNull()) {
				meta = StringValue::Get(mv);
			}
		}
	} else if (arguments.size() != base_args) {
		throw BinderException("ai_function_with_embed: expected 1 tree + 3*" + std::to_string(leaf_count) +
		                      " texts (+ optional meta_str), got " + std::to_string(arguments.size()) + " args");
	}
	// Speculative node: a row whose estimated P(pass) reaches the gate passes through unevaluated.
	const double threshold = AI_SPECULATIVE_THRESHOLD;
	shared_ptr<AIFilterTreeNode> tree = std::move(parsed);
	auto wrappers = AILeafWrappers(input.GetClientContext(), meta, leaf_count);
	return make_uniq<AIFilterWithEmbedBindData>(std::move(tree), tree_str, leaf_count, speculative, threshold,
	                                            /*limit=*/-1, std::move(meta), std::move(wrappers));
}

static unique_ptr<FunctionData> AIFilterWithEmbedBind(BindScalarFunctionInput &input) {
	return AIFilterWithEmbedBindImpl(input, /*speculative=*/false);
}

static unique_ptr<FunctionData> AISpeculativeFilterWithEmbedBind(BindScalarFunctionInput &input) {
	return AIFilterWithEmbedBindImpl(input, /*speculative=*/true);
}

//! Extract a row's embedding (LIST(FLOAT)) into a float vector.
static vector<float> ReadEmbedding(Vector &col, idx_t row) {
	vector<float> vec;
	Value v = col.GetValue(row);
	if (v.IsNull()) {
		return vec;
	}
	auto &children = ListValue::GetChildren(v);
	vec.reserve(children.size());
	for (auto &c : children) {
		vec.push_back(c.IsNull() ? 0.0f : c.GetValue<float>());
	}
	return vec;
}

//! Cosine similarity of two equal-length embeddings (0 if degenerate).
static float CosineSim(const vector<float> &a, const vector<float> &b) {
	if (a.empty() || a.size() != b.size()) {
		return 0.0f;
	}
	double dot = 0.0, na = 0.0, nb = 0.0;
	for (idx_t i = 0; i < a.size(); i++) {
		dot += static_cast<double>(a[i]) * b[i];
		na += static_cast<double>(a[i]) * a[i];
		nb += static_cast<double>(b[i]) * b[i];
	}
	if (na <= 0.0 || nb <= 0.0) {
		return 0.0f;
	}
	return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

//! MLP input feature for a (row, leaf): [predicate_emb, input_emb, cos_sim(predicate, input)].
//! Empty if either embedding is missing.
static vector<float> BuildPredicateFeature(const vector<float> &pred_emb, const vector<float> &input_emb) {
	vector<float> feat;
	if (pred_emb.empty() || input_emb.empty()) {
		return feat;
	}
	feat.reserve(pred_emb.size() + input_emb.size() + 1);
	feat.insert(feat.end(), pred_emb.begin(), pred_emb.end());
	feat.insert(feat.end(), input_emb.begin(), input_emb.end());
	feat.push_back(CosineSim(pred_emb, input_emb));
	return feat;
}

// An image leaf's input carries the ai_image sentinel. With the default embeddings server (CLIP for images)
// the leaf gets the same pair feature as a text leaf (predicate text x image, one joint space) and the MLP learns
// its selectivity; the notes below describe the fallback when images cannot be embedded (text-only
// server, ai_embed_images=false), detected per leaf via AITextHasImage(prompt) + AIEmbedImagesSupported():
//   - selectivity: unknowable -> a neutral 0.5 (max-entropy), so the ordering is decided by cost alone,
//     deterministically. In particular the dedup-domain cost scaling (see AIFilterEvaluateBatch) makes a
//     leaf with few distinct prompts run before one with many -- "smaller dedup-domain first" -- instead
//     of letting a fictitious per-row estimate flip the order. Order-independent -> the boolean result is
//     unchanged; only which calls short-circuit (cost) differs.
//   - cost: a vision call costs far more than the short sentinel's text length implies. Estimate it from the
//     actual image size on disk (below) rather than the sentinel string, so the DP defers big images but can
//     still run a tiny thumbnail before a long-document text leaf.

// Byte size of an image ref: a local file's size on disk (metadata only -- seek to end, no content read),
// or the decoded length of a data: URI; -1 if unknown (a remote http(s) URL we won't fetch just to cost it).
static double AIImageRefBytes(const string &ref) {
	if (ref.compare(0, 5, "data:") == 0) {
		const auto comma = ref.find(',');
		return comma == string::npos ? -1.0 : static_cast<double>(ref.size() - comma - 1) * 3.0 / 4.0;
	}
	if (ref.compare(0, 7, "http://") == 0 || ref.compare(0, 8, "https://") == 0) {
		return -1.0;
	}
	std::ifstream f(ref, std::ios::binary | std::ios::ate);
	return f.good() ? static_cast<double>(f.tellg()) : -1.0;
}

// Token-cost estimate for an image leaf: the text portion of the prompt (predicate etc.) via the normal cost
// model, plus each embedded image mapped from its byte size to tokens. The byte->token map approximates
// OpenAI vision billing (per-image base + a per-tile term) using file size as a monotonic proxy for
// resolution -- exact tiles need pixel dimensions, but size needs no decode and orders images correctly.
static double AIImageLeafCost(const string &prompt) {
	constexpr double base = 85.0;             // OpenAI per-image base tokens
	constexpr double bytes_per_token = 350.0; // ~high-detail tiles vs. file size
	constexpr double url_tokens = 765.0;      // nominal ~1MP when the size is unknown (remote URL)
	constexpr double max_tokens = 8192.0;     // clamp a pathologically large file
	string text;
	double image_tokens = 0.0;
	for (idx_t i = 0; i < prompt.size();) {
		if (prompt[i] == AI_IMAGE_OPEN) {
			const auto close = prompt.find(AI_IMAGE_CLOSE, i + 1);
			if (close == string::npos) {
				text += prompt.substr(i);
				break;
			}
			const double bytes = AIImageRefBytes(prompt.substr(i + 1, close - i - 1));
			const double t = bytes >= 0.0 ? base + bytes / bytes_per_token : url_tokens;
			image_tokens += t < max_tokens ? t : max_tokens;
			i = close + 1;
		} else {
			text.push_back(prompt[i]);
			i++;
		}
	}
	return AIEstimatePromptCost(text) + image_tokens;
}

// Append one ai_function_with_embed args DataChunk (column layout [tree_str, prompt_0..n-1, pred_0..n-1,
// input_0..n-1]) onto the growing per-(row, leaf) text/cost vectors. Rows with any NULL leaf prompt are
// marked invalid. Append semantics let the streaming operator accumulate a whole buffered join output
// across many chunks before one AIFilterEvaluateBatch call; the scalar path appends a single chunk.
static void AIFilterAppendArgs(DataChunk &args, idx_t n, vector<vector<string>> &prompt,
                               vector<vector<string>> &pred_text, vector<vector<string>> &input_text,
                               vector<vector<double>> &cost, vector<char> &row_valid) {
	const idx_t count = args.size();
	const idx_t prompt_base = 1;
	const idx_t pred_text_base = 1 + n;      // predicate text (constant question part) per leaf
	const idx_t input_text_base = 1 + 2 * n; // input text (row-varying document part) per leaf
	const idx_t base = prompt.size();
	prompt.resize(base + count, vector<string>(n));
	pred_text.resize(base + count, vector<string>(n));
	input_text.resize(base + count, vector<string>(n));
	cost.resize(base + count, vector<double>(n, 1.0));
	row_valid.resize(base + count, 1);
	for (idx_t l = 0; l < n; l++) {
		auto &pcol = args.data[prompt_base + l];
		auto &ptcol = args.data[pred_text_base + l];
		auto &itcol = args.data[input_text_base + l];
		for (idx_t row = 0; row < count; row++) {
			const idx_t r = base + row;
			Value pv = pcol.GetValue(row);
			if (pv.IsNull()) {
				row_valid[r] = 0;
			} else {
				prompt[r][l] = StringValue::Get(pv);
				cost[r][l] =
				    AITextHasImage(prompt[r][l]) ? AIImageLeafCost(prompt[r][l]) : AIEstimatePromptCost(prompt[r][l]);
			}
			Value ptv = ptcol.GetValue(row);
			if (!ptv.IsNull()) {
				pred_text[r][l] = StringValue::Get(ptv);
			}
			Value itv = itcol.GetValue(row);
			if (!itv.IsNull()) {
				input_text[r][l] = StringValue::Get(itv);
			}
		}
	}
}

// Per-leaf kind + comparison for a generalized ai_function_with_embed node. kind: F=ai_filter (boolean),
// C=ai_classify (string == val), S=ai_score (double <op> val), M=ai_complete (raw string == val). op:
// 'e'=,'n'!=,'g'>,'l'<,'G'>=,'L'<= ; 'i'=IN / 'I'=NOT IN (val is a ','-joined set). Unused for F. Empty
// meta -> every leaf is F (unchanged behavior).
struct AILeafMeta {
	char kind = 'F';
	char op = 0;
	string val;
	bool has_range = false; // ranged ai_score (kind S): clamp to [lo,hi] and round (is_int) before comparing
	double lo = 0;
	double hi = 1;
	bool is_int = false;
	//! Wrapped leaf (op 'w'): the boolean expression over the answer (column 0 of `wrapper_type`), owned by
	//! the node's bind data.
	const Expression *wrapper = nullptr;
	LogicalType wrapper_type;
};

// The type the wrapper reads its answer as: its column-0 reference's type.
static bool AIWrapperRefType(const Expression &expr, LogicalType &out) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		out = expr.GetReturnType();
		return true;
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!found) {
			found = AIWrapperRefType(child, out);
		}
	});
	return found;
}

// Parse a meta_str ("F;C:eq:health;S:gt:0.5;M:wrap:<b64>") into `n` per-leaf entries (one per ';').
// Missing/short -> F. A `wrap` token takes its deserialized wrapper from `wrappers` (leaf-indexed).
static vector<AILeafMeta> AIParseLeafMeta(const string &meta, idx_t n, const vector<shared_ptr<Expression>> &wrappers) {
	vector<AILeafMeta> out(n);
	if (meta.empty()) {
		return out;
	}
	idx_t leaf = 0;
	size_t i = 0;
	while (leaf < n && i <= meta.size()) {
		const size_t semi = meta.find(';', i);
		const string tok = meta.substr(i, (semi == string::npos ? meta.size() : semi) - i);
		if (!tok.empty() && tok != "F") {
			const size_t c1 = tok.find(':');
			const size_t c2 = (c1 == string::npos) ? string::npos : tok.find(':', c1 + 1);
			out[leaf].kind = tok[0];
			if (c1 != string::npos && c2 != string::npos) {
				const string op = tok.substr(c1 + 1, c2 - c1 - 1);
				out[leaf].val = tok.substr(c2 + 1);
				out[leaf].op = op == "ne"     ? 'n'
				               : op == "gt"   ? 'g'
				               : op == "lt"   ? 'l'
				               : op == "ge"   ? 'G'
				               : op == "le"   ? 'L'
				               : op == "in"   ? 'i'
				               : op == "ni"   ? 'I'
				               : op == "wrap" ? 'w'
				                              : 'e';
				if (out[leaf].op == 'w') {
					if (leaf >= wrappers.size() || !wrappers[leaf] ||
					    !AIWrapperRefType(*wrappers[leaf], out[leaf].wrapper_type)) {
						throw InternalException("ai_function_with_embed: wrapped leaf " + std::to_string(leaf) +
						                        " has no bound wrapper");
					}
					out[leaf].wrapper = wrappers[leaf].get();
				}
				// Ranged ai_score encodes "cmpval@lo,hi,i|d"; split off the range for clamp/round.
				if (out[leaf].kind == 'S') {
					const size_t at = out[leaf].val.find('@');
					if (at != string::npos) {
						const string range = out[leaf].val.substr(at + 1);
						out[leaf].val = out[leaf].val.substr(0, at);
						const size_t k1 = range.find(',');
						const size_t k2 = (k1 == string::npos) ? string::npos : range.find(',', k1 + 1);
						if (k1 != string::npos && k2 != string::npos) {
							out[leaf].lo = std::atof(range.substr(0, k1).c_str());
							out[leaf].hi = std::atof(range.substr(k1 + 1, k2 - k1 - 1).c_str());
							out[leaf].is_int = range[k2 + 1] == 'i';
							out[leaf].has_range = true;
						}
					}
				}
			}
		}
		leaf++;
		if (semi == string::npos) {
			break;
		}
		i = semi + 1;
	}
	return out;
}

// Evaluate one leaf: send its (rewrite-baked, scalar-exact) prompt as the leaf's AI function, then apply
// the comparison -> the leaf boolean. `ok` is set false on a failed/unparseable call (leaf -> false).
// `pred_text` (F leaves): the leaf's constant question part, from the node's feature columns.
static AIRequest AILeafRequest(const AILeafMeta &m, const string &prompt, const string *pred_text = nullptr) {
	AIRequest req;
	req.prompt = prompt;
	if (m.kind == 'F') {
		req.system_prompt = AIFilterSystemPrompt();
		req.json_schema = AIFilterSchema();
		req.schema_name = "ai_filter";
		AIMarkFilterQuestion(req, pred_text ? *pred_text : string());
	} else if (m.kind == 'C') {
		req.system_prompt = AI_CLASSIFY_SYSTEM_PROMPT;
		req.json_schema = STRING_RESULT_SCHEMA;
		req.schema_name = "ai_classify";
		// The rewrite bakes the scalar's exact "Categories: a, b\n\nInput:\n<input>\n\n..." prompt
		// (ai_filter_tree_build); recover the option list and input from that fixed frame so the
		// leaf asks the same Choice the scalar would.
		static const string kCats = "Categories: ";
		static const string kInput = "\n\nInput:\n";
		static const string kTail = "\n\nRespond with exactly one of the listed categories, verbatim.";
		const auto input_at = prompt.find(kInput);
		if (prompt.compare(0, kCats.size(), kCats) == 0 && input_at != string::npos && prompt.size() >= kTail.size() &&
		    prompt.compare(prompt.size() - kTail.size(), kTail.size(), kTail) == 0) {
			vector<std::pair<string, string>> options;
			for (auto &cat : StringUtil::Split(prompt.substr(kCats.size(), input_at - kCats.size()), ", ")) {
				options.emplace_back(cat, string());
			}
			const idx_t in_begin = input_at + kInput.size();
			AIMarkClassifyQuestion(req, prompt.substr(in_begin, prompt.size() - kTail.size() - in_begin),
			                       std::move(options), string());
		}
	} else if (m.kind == 'S') {
		req.system_prompt = AI_SCORE_SYSTEM_PROMPT;
		req.json_schema = NUMBER_RESULT_SCHEMA;
		req.schema_name = "ai_score";
		// The baked ranged prompt is "Provide a score ...\n\nScoring criteria: <criteria>\n\nInput:\n<input>"
		// (AIScoreUserPrompt); split it back so a Score can take the input as its state.
		static const string kCrit = "\n\nScoring criteria: ", kIn = "\n\nInput:\n";
		const auto crit_at = prompt.find(kCrit);
		const auto in_at = crit_at == string::npos ? string::npos : prompt.find(kIn, crit_at + kCrit.size());
		if (m.has_range && in_at != string::npos) {
			AIMarkScoreQuestion(req, prompt.substr(in_at + kIn.size()),
			                    prompt.substr(crit_at + kCrit.size(), in_at - crit_at - kCrit.size()), m.lo, m.hi,
			                    m.is_int);
		}
	} else { // 'M' -> ai_complete: raw completion, no forced schema
		req.system_prompt = AI_COMPLETE_SYSTEM_PROMPT;
		req.schema_name = "ai_complete";
	}
	return req;
}

//! Evaluate a wrapped leaf's wrapper over the answer: a one-row chunk holding the answer as the wrapper's
//! column 0. A failed or unparseable call is a NULL answer, exactly the scalar's, so `IS NULL` and friends
//! keep their meaning; a NULL result (e.g. a TRY_CAST that failed on the answer) is FALSE, as in a filter.
static bool AIApplyWrapper(ClientContext &context, const AILeafMeta &m, const Value &answer) {
	DataChunk row;
	row.Initialize(Allocator::DefaultAllocator(), {m.wrapper_type});
	row.SetChildCardinality(1);
	row.data[0].SetValue(0, answer.DefaultCastAs(m.wrapper_type));
	ExpressionExecutor executor(context, *m.wrapper);
	Vector out(LogicalType::BOOLEAN);
	executor.ExecuteExpression(row, out);
	const Value v = out.GetValue(0);
	return !v.IsNull() && BooleanValue::Get(v);
}

//! Fold a leaf's raw response content into the leaf's boolean outcome (comparison or wrapper applied).
static bool AILeafOutcome(ClientContext &context, const AILeafMeta &m, const string &content, bool &ok) {
	if (m.kind == 'F') {
		bool v = false;
		ok = AIReadFilterVerdict(content, v);
		if (m.wrapper) {
			const bool wrapped = AIApplyWrapper(context, m, ok ? Value::BOOLEAN(v) : Value(LogicalType::BOOLEAN));
			ok = true; // the wrapper decided: a valid verdict even over a missing or unparseable answer
			return wrapped;
		}
		return ok && v;
	}
	if (m.kind == 'S') {
		double d = 0;
		ok = AIParseDoubleField(content, "result", d);
		if (ok && m.has_range) { // clamp to [lo,hi] then round -- matches the ranged ai_score scalar before comparing
			d = d < m.lo ? m.lo : (d > m.hi ? m.hi : d);
			if (m.is_int) {
				d = static_cast<double>(std::llround(d));
			}
		}
		if (m.wrapper) {
			const bool wrapped = AIApplyWrapper(context, m, ok ? Value::DOUBLE(d) : Value(LogicalType::DOUBLE));
			ok = true; // the wrapper decided: a valid verdict even over a missing or unparseable answer
			return wrapped;
		}
		if (!ok) {
			return false;
		}
		const double v = std::atof(m.val.c_str());
		switch (m.op) {
		case 'g':
			return d > v;
		case 'l':
			return d < v;
		case 'G':
			return d >= v;
		case 'L':
			return d <= v;
		case 'n':
			return d != v;
		default:
			return d == v;
		}
	}
	// C (classify) or M (complete): compare the string result.
	string s;
	if (m.kind == 'C') {
		ok = AIParseStringField(content, "result", s);
	} else {
		s = content; // ai_complete returns the raw content
		ok = true;
	}
	if (m.wrapper) {
		const bool wrapped = AIApplyWrapper(context, m, ok ? Value(s) : Value(LogicalType::VARCHAR));
		ok = true; // the wrapper decided: a valid verdict even over a missing or unparseable answer
		return wrapped;
	}
	if (!ok) {
		return false;
	}
	if (m.op == 'i' || m.op == 'I') {
		// Set membership (IN / NOT IN): m.val is the ','-joined set; evaluate classify/complete ONCE, then
		// test the result against the set (vs the desugared OR-of-equalities that would re-run the call).
		bool in_set = false;
		size_t start = 0;
		while (true) {
			const size_t comma = m.val.find(',', start);
			const string member = m.val.substr(start, (comma == string::npos ? m.val.size() : comma) - start);
			if (s == member) {
				in_set = true;
				break;
			}
			if (comma == string::npos) {
				break;
			}
			start = comma + 1;
		}
		return m.op == 'i' ? in_set : !in_set;
	}
	const bool eq = (s == m.val);
	return m.op == 'n' ? !eq : eq;
}

static bool AIEvalLeaf(ClientContext &context, const AILeafMeta &m, const string &prompt, const string &query_text,
                       bool &ok, const string *prefix = nullptr, idx_t expected_reuse = 0,
                       const string *pred_text = nullptr, const AIConfig *config = nullptr) {
	vector<AIRequest> one;
	if (prefix && !prefix->empty()) {
		// Build from the full text (a Choice leaf recovers its options from the baked frame), then
		// split: the request's prompt is the SUFFIX when a prefix rides along.
		one.push_back(AILeafRequest(m, *prefix + prompt, pred_text));
		one[0].prompt = prompt;
		one[0].prompt_prefix = *prefix;
		one[0].expected_reuse = expected_reuse; // fan-out hint for the client's write policy
	} else {
		one.push_back(AILeafRequest(m, prompt, pred_text));
	}
	auto r = AIBatchComplete(one, query_text, /*force_fixed=*/true, config);
	ok = !r.empty() && r[0].success;
	if (!ok) {
		if (m.wrapper) {
			// A failed call is a NULL answer to the wrapper, whose result is the leaf's (valid) verdict: IS NULL
			// is true for it, `= 'yes'` is false. Without a wrapper the verdict stays unknown (ok false).
			ok = true;
			return AIApplyWrapper(context, m, Value(m.wrapper_type));
		}
		return false;
	}
	return AILeafOutcome(context, m, r[0].content, ok);
}

//! Cache-only leaf evaluation: fold the leaf from the query-local response cache; NEVER issues a
//! call. Returns false on a cache miss (leaf stays unknown). Backs the LIMIT stand-down's free
//! pruning: a stood-down speculative node may drop rows whose answer is already known, but must
//! never spend a call the k-bounded evaluation above may not need.
static bool AIEvalLeafCached(ClientContext &context, const AILeafMeta &m, const string &prompt,
                             const string &query_text, bool &value, const string *pred_text = nullptr,
                             const AIConfig *config = nullptr) {
	AIResult cached;
	if (!AICacheProbe(AILeafRequest(m, prompt, pred_text), query_text, cached, config) || !cached.success) {
		return false;
	}
	bool ok = false;
	value = AILeafOutcome(context, m, cached.content, ok);
	return ok;
}

// Cold-MLP warm-up: pick one still-UNKNOWN leaf of `row` by a SEEDED hash of (row, #known leaves). The
// choice depends only on the row and its progress -- never on which worker runs it or when -- so a cold
// start is reproducible run-to-run (unlike letting the untrained MLP's noisy P(pass) drive the DP). Splits
// label collection across leaves rather than always probing leaf 0 first. Returns n if nothing is unknown.
static idx_t AIWarmupPickLeaf(idx_t row, const vector<AITriState> &leaf_values, idx_t n) {
	idx_t unknown = 0;
	for (idx_t l = 0; l < n; l++) {
		if (leaf_values[l] == AITriState::TRI_UNKNOWN) {
			unknown++;
		}
	}
	if (unknown == 0) {
		return n;
	}
	// splitmix64 over (row, #known) -> pick the k-th still-unknown leaf.
	uint64_t h = static_cast<uint64_t>(row) + 0x9E3779B97F4A7C15ull * (static_cast<uint64_t>(n - unknown) + 1);
	h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
	h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
	h ^= h >> 31;
	idx_t pick = static_cast<idx_t>(h % unknown);
	for (idx_t l = 0; l < n; l++) {
		if (leaf_values[l] == AITriState::TRI_UNKNOWN) {
			if (pick == 0) {
				return l;
			}
			pick--;
		}
	}
	return n; // unreachable when unknown > 0
}

// Evaluate `count` rows of a (speculative) ai_function_with_embed node against the live model: shared
// worker pool, row-tuple dedup, single-flight per leaf prompt, speculative gate + inline training.
// Writes out_result[row] (fanned out to duplicates; 0/1) for every valid row. Callers pass the already
// split per-(row, leaf) prompt/predicate/input text + cost + validity, so this serves both the scalar
// function (one DataChunk) and the streaming dedup operator (a whole buffered join output at once).
static void AIFilterEvaluateBatch(ClientContext &context, const AIFilterWithEmbedBindData &bind_data, idx_t count,
                                  const vector<vector<string>> &prompt, const vector<vector<string>> &pred_text,
                                  const vector<vector<string>> &input_text, const vector<vector<double>> &cost,
                                  const vector<char> &row_valid, const string &query_text, vector<char> &out_result,
                                  int64_t limit_override = -2, const vector<idx_t> *out_weights = nullptr) {
	const AIConfig batch_config = AIConfigForContext(context); // this connection's endpoint/model/key for the batch
	const auto &tree = *bind_data.tree;
	const idx_t n = bind_data.leaf_count;
	// LIMIT push-down: stop once this many rows PASS (-1 = no limit). limit_override != -2 substitutes an
	// external k (the AI-dedup operator passes k here); out_weights[r] (when set) is row r's fan-out count, so
	// a passing representative counts for all the OUTPUT rows it broadcasts to -- the count-accurate early-stop.
	const int64_t limit = (limit_override != -2) ? limit_override : bind_data.limit;
	// per-leaf function + comparison / wrapper
	const vector<AILeafMeta> leaf_meta = AIParseLeafMeta(bind_data.meta, n, bind_data.wrappers);
	// Cache/single-flight key = prompt + this per-leaf comparison, so two leaves that share a prompt but
	// differ in comparison (e.g. ai_classify(x)='a' AND ai_classify(x)='b') never collide on one boolean.
	vector<string> leaf_key(n);
	for (idx_t l = 0; l < n; l++) {
		if (leaf_meta[l].kind != 'F') {
			leaf_key[l] = string("\x1f") + leaf_meta[l].op + leaf_meta[l].val;
		}
	}

	// Training overlaps the blocking LLM batch call unless explicitly disabled.
	const bool train_enabled = std::getenv("DUCKDB_AI_NO_TRAIN") == nullptr;
	// MLP feature: factored [pred_emb, input_emb, cos_sim] (default; document embedded once, shared
	// across leaves) vs a single embed(full prompt). AI_MLP_FEATURE=prompt selects the latter, for
	// A/B comparison. The MLP lazily adapts to whichever input width it first sees.
	const char *feat_env = std::getenv("AI_MLP_FEATURE");
	const bool feature_prompt_mode = feat_env && string(feat_env) == "prompt";
	// Features (one embed per leaf per row) only matter where a prediction can change behaviour: ordering
	// the leaves of a tree with more than one, or a speculative gate. A one-leaf plain tree has nothing to
	// order, so it embeds nothing and records no training example (MOVIE's LIMIT queries spent more time in
	// these per-row embeds than in the model calls they could never influence).
	const bool needs_features = n > 1 || bind_data.speculative;
	// Fixed LLM batch size (default 20, matching Sembench): each round's evaluations are sent in
	// batches of this many and one MLP training step overlaps each batch, so the training cadence is
	// ~1 step per `batch_size` calls. Reuses the fixed-concurrency knob so a batch runs in one wave.
	idx_t batch_size = 20;
	if (const char *bs = std::getenv("AI_MAX_CONCURRENCY")) {
		const int v = std::atoi(bs);
		if (v > 0) {
			batch_size = static_cast<idx_t>(v);
		}
	}
	// Shared worker-pool routing: `batch_size` (== fixed concurrency) workers pull undetermined rows
	// from a queue and keep the queue full -- like the baseline's single shared AIBatchComplete queue,
	// so there are no batch-boundary straggler waits. Each worker JIT-predicts p_true against the live
	// model, DP-chooses the row's next leaf, makes one LLM call, applies the result, and re-queues the
	// row if it needs another leaf. Training runs inline every `batch_size` completions (single-flight),
	// so the model keeps improving the predictions of rows scheduled later. Order-independent -> the
	// boolean result is unchanged; only which calls happen (cost) and their scheduling (latency) differ.
	vector<vector<AITriState>> leaf_values(count, vector<AITriState>(n, AITriState::TRI_UNKNOWN));
	vector<char> row_result(count, 0);

	// De-duplicate rows by their full prompt tuple: rows with identical leaf prompts are
	// evaluation-equivalent (the boolean result depends only on the LLM answers to those prompts), so
	// we evaluate ONE representative per distinct tuple and fan its result out. This matters when the
	// node runs on a duplicated input -- e.g. a pulled-up filter over a join fan-out (2450 rows / 50
	// distinct docs) -- where processing every row would re-embed, re-predict, and (racing on the
	// response cache under concurrency) re-issue the same LLM call many times over.
	vector<idx_t> row_rep(count, 0);
	vector<idx_t> reps; // distinct representative row indices (valid rows only)
	{
		std::unordered_map<string, idx_t> rep_of_key;
		for (idx_t row = 0; row < count; row++) {
			if (!row_valid[row]) {
				continue;
			}
			string key;
			for (idx_t l = 0; l < n; l++) {
				key += prompt[row][l];
				key.push_back('\x1f');
			}
			auto it = rep_of_key.find(key);
			if (it == rep_of_key.end()) {
				rep_of_key.emplace(std::move(key), row);
				reps.push_back(row);
				row_rep[row] = row;
			} else {
				row_rep[row] = it->second;
			}
		}
	}
	const idx_t rep_count = reps.size();

	// Per-leaf dedup-domain cost scaling. Identical prompts across rows resolve with ONE call (the
	// single-flight prompt cache plus the fold step below), so a leaf's expected marginal cost per row is
	// its per-call cost times distinct_prompts/rep_count. A leaf whose inputs come from a join's small side
	// (e.g. 500 distinct images under a 3000-pair fan-out) is that much cheaper than its per-call cost
	// suggests, and the DP evaluates it first: the deterministic "smaller dedup-domain first" rule, which
	// decides the order whenever selectivity is unknowable (image leaves, all p = 0.5).
	vector<double> domain_frac(n, 1.0);
	if (rep_count > 0) {
		std::hash<string> hasher;
		std::unordered_set<uint64_t> distinct;
		for (idx_t l = 0; l < n; l++) {
			distinct.clear();
			for (const auto rep : reps) {
				distinct.insert(hasher(prompt[rep][l]));
			}
			domain_frac[l] = static_cast<double>(distinct.size()) / static_cast<double>(rep_count);
		}
	}

	std::mutex q_mutex;
	std::condition_variable q_cv;
	std::deque<idx_t> workq(reps.begin(), reps.end()); // only representatives are scheduled
	std::atomic<idx_t> resolved {0};                   // resolved representatives
	std::atomic<uint64_t> completions {0};
	// Shared cold-MLP warm-up latch. The process-global selectivity model is "warm" once it has completed a
	// training step. BOTH warm-up paths consult this ONE latch -- the reorder node's seeded-random leaf pick
	// and the speculative gate's forced-evaluate -- so neither re-warms a model that the other path, or an
	// earlier chunk/query in this process, already trained. Starts warm when already trained; latches on the
	// first train step observed during this batch. AI_REORDER_NO_WARMUP forces it off (for A/B).
	const bool warmup_disabled = std::getenv("AI_REORDER_NO_WARMUP") != nullptr;
	std::atomic<bool> mlp_warm {warmup_disabled || AISelectivityModel::Global().Stats().train_steps > 0};
	// LIMIT push-down: once `limit` rows have PASSED, stop the whole pool -- the LIMIT above keeps only k,
	// so the still-unevaluated rows can stay false. A row that passes calls note_pass(); the workers and the
	// trainer bail out as soon as limit_hit is set (in-flight rows just finish their current leaf).
	std::atomic<int64_t> passed {0};
	std::atomic<bool> limit_hit {false};
	const idx_t nworkers_hint = rep_count == 0 ? 0 : (batch_size < rep_count ? batch_size : rep_count);
	// Embedding prefetch (overlap CPU-side embedding with in-flight LLM latency): a dedicated
	// thread pre-embeds queued rows' unknown leaves into this cache; workers consult it before
	// embedding inline. Keyed (row, leaf) -> feature vector; empty vector = embed failed.
	std::mutex feat_mutex;
	std::unordered_map<uint64_t, vector<float>> feat_cache;
	// Overlap accounting: EMA of per-leaf LLM latency and per-row embed latency (microseconds).
	// The prefetcher bounds its lead so embeds ride entirely inside LLM waits: during one LLM
	// request the pool consumes ~C rows, so a lead of C*embed/llm rows (floored at C) is all the
	// overlap ever needs; anything deeper risks embedding rows that resolve via prompt-cache dups.
	std::atomic<int64_t> llm_lat_us {0};
	std::atomic<int64_t> embed_lat_us {0};
	auto ema_update = [](std::atomic<int64_t> &ema, int64_t sample) {
		const int64_t old = ema.load(std::memory_order_relaxed);
		ema.store(old == 0 ? sample : (old * 7 + sample) / 8, std::memory_order_relaxed);
	};
	auto feat_key = [n](idx_t r, idx_t l) {
		return static_cast<uint64_t>(r) * n + l;
	};
	auto note_pass = [&](idx_t r) {
		const int64_t w = out_weights ? static_cast<int64_t>((*out_weights)[r]) : 1; // fan-out count (1 if unweighted)
		if (limit >= 0 && passed.fetch_add(w) + w >= limit) {
			std::lock_guard<std::mutex> lock(q_mutex);
			limit_hit.store(true);
			q_cv.notify_all();
		}
	};

	// Apply leaf `lf`'s boolean result to row `r`, then resolve the row (tree determined) or re-queue it
	// for its next leaf. Called both by a worker for its own row and by a prompt's fetcher for the rows
	// that parked on it. Thread-safe: leaf_values[r]/row_result[r] have a single owner at a time.
	// Idempotent resolution: with the pull-ahead below, one row can receive results from BOTH a
	// parked leaf and a pulled-ahead leaf; only the first tree resolution may count it.
	std::unique_ptr<std::atomic<char>[]> row_done(new std::atomic<char>[count]());
	auto route_row = [&](idx_t r, idx_t lf, bool val) {
		if (row_done[r].load()) {
			return; // already resolved (a later-arriving parked result changes nothing)
		}
		leaf_values[r][lf] = val ? AITriState::TRI_TRUE : AITriState::TRI_FALSE;
		const auto tv = AIFilterTreeEval(tree, leaf_values[r]);
		if (tv != AITriState::TRI_UNKNOWN) {
			if (row_done[r].exchange(1)) {
				return; // another thread resolved concurrently
			}
			row_result[r] = (tv == AITriState::TRI_TRUE) ? 1 : 0;
			if (row_result[r]) {
				note_pass(r);
			}
			if (resolved.fetch_add(1) + 1 >= rep_count) {
				std::lock_guard<std::mutex> lock(q_mutex);
				q_cv.notify_all(); // last representative done -> wake every worker to exit
			}
		} else {
			std::lock_guard<std::mutex> lock(q_mutex);
			workq.push_back(r); // needs another leaf
			q_cv.notify_one();
		}
	};

	// Single-flight per leaf prompt: only ONE LLM call per distinct prompt string, even across the
	// concurrent workers (the query response cache alone races -- 20 workers can all miss the same
	// prompt before any writes it back, re-issuing it). Distinct prompts still run fully in parallel.
	// Handles what the row-tuple dedup can't: two DISTINCT rows sharing a leaf prompt, e.g. a pulled-up
	// H(a) AND S(b) over a join where each a's prompt recurs across its many b partners. Non-blocking:
	// a worker that hits an in-flight prompt PARKS its (row, leaf) and grabs the next row rather than
	// blocking; the fetcher routes all parked rows when its result lands, so concurrency never stalls.
	struct PromptEval {
		bool ready = false;
		bool value = false;
		vector<std::pair<idx_t, idx_t>> waiters; // (row, leaf) parked on this in-flight prompt
	};
	std::mutex pc_mutex;
	std::unordered_map<string, PromptEval> prompt_cache;
	std::mutex error_mutex;
	ErrorData first_error; // the first exception a worker's leaf evaluation raised, rethrown after the join

	auto worker_body = [&]() {
		while (true) {
			idx_t row;
			{
				std::unique_lock<std::mutex> lock(q_mutex);
				q_cv.wait(lock, [&] { return !workq.empty() || resolved.load() >= rep_count || limit_hit.load(); });
				if (limit_hit.load() || workq.empty()) {
					return; // LIMIT reached, or all representatives resolved
				}
				row = workq.front();
				workq.pop_front();
			}
			// Fold any leaf whose prompt is already answered in the shared cache into this row's tree
			// state. Duplicated inputs (a pulled-up filter over a join fan-out) reuse leaf prompts across
			// rows, so a leaf resolved for one row is known for all. This lets the tree short-circuit
			// deterministically -- if a known leaf makes the boolean determined we resolve with ZERO
			// LLM calls and ZERO embeds -- and otherwise skips embedding/predicting the already-known
			// leaves (the embed/DP steps below only touch leaves still marked UNKNOWN).
			{
				std::lock_guard<std::mutex> lk(pc_mutex);
				for (idx_t l = 0; l < n; l++) {
					if (leaf_values[row][l] != AITriState::TRI_UNKNOWN) {
						continue;
					}
					auto it = prompt_cache.find(prompt[row][l] + leaf_key[l]);
					if (it != prompt_cache.end() && it->second.ready) {
						leaf_values[row][l] = it->second.value ? AITriState::TRI_TRUE : AITriState::TRI_FALSE;
					}
				}
			}
			{
				const auto folded = AIFilterTreeEval(tree, leaf_values[row]);
				if (folded != AITriState::TRI_UNKNOWN) {
					row_result[row] = (folded == AITriState::TRI_TRUE) ? 1 : 0;
					if (row_result[row]) {
						note_pass(row);
					}
					if (resolved.fetch_add(1) + 1 >= rep_count) {
						std::lock_guard<std::mutex> lock(q_mutex);
						q_cv.notify_all();
					}
					continue; // determined purely from cached leaf values
				}
			}
			if (bind_data.speculative) {
				// CACHE-ONLY, always. A speculative leaf prunes rows whose verdict is ALREADY
				// KNOWN and does nothing else: no embedding, no MLP, no call. That is free and it
				// cannot mispredict, since it only drops a row whose conjunction is known FALSE.
				//
				// It used to embed each row, run the selectivity model and pass rows it predicted
				// would survive, which costs an embed + predict PER ROW where the region evaluates
				// a whole wave at once -- and it only pays when a real fan-out sits between this
				// leaf and the predicate above it. Dedup usually removes that fan-out first: the
				// duplicates a join creates share a prompt, so the local cache and the in-flight
				// registry already collapse them. Measured on a CTE-feeding-a-join shape: 26 calls
				// and 116 rows either way, 9.11s with the predict path against 1.57s without it.
				// The prediction bought nothing and cost 5.8x the latency.
				//
				// (This is what the LIMIT stand-down already did; it is now the only behaviour.)
				for (idx_t l = 0; l < n; l++) {
					if (leaf_values[row][l] != AITriState::TRI_UNKNOWN) {
						continue;
					}
					bool v = false;
					if (AIEvalLeafCached(context, leaf_meta[l], prompt[row][l], query_text, v, &pred_text[row][l],
					                     &batch_config)) {
						leaf_values[row][l] = v ? AITriState::TRI_TRUE : AITriState::TRI_FALSE;
					}
				}
				const auto folded = AIFilterTreeEval(tree, leaf_values[row]);
				row_result[row] = (folded == AITriState::TRI_FALSE) ? 0 : 1;
				if (row_result[row]) {
					note_pass(row);
				}
				if (resolved.fetch_add(1) + 1 >= rep_count) {
					std::lock_guard<std::mutex> lock(q_mutex);
					q_cv.notify_all();
				}
				continue;
			}
			vector<double> p_row(n, 0.5);
			vector<vector<float>> feats(n);
			// Shared cold-MLP latch update: once the process-global model has trained a step, warm-up stops --
			// the seeded-random leaf pick (both nodes) here, and the speculative forced-evaluate below.
			// Queried only while still cold, then latched, so an already-warm model is never re-warmed.
			if (!mlp_warm.load() && AISelectivityModel::Global().Stats().train_steps > 0) {
				mlp_warm.store(true);
			}
			const bool cold = !mlp_warm.load();
			// Cold-MLP warm-up (BOTH the reorder and speculative nodes): while cold, pick a SEEDED-random
			// unknown leaf and embed ONLY it -- skipping embed-all + JIT-predict + DP below. Spreads the first
			// labels evenly across the leaves (a biased cold model would keep probing one leaf) and is
			// reproducible run-to-run. The speculative gate is already skipped while cold (it force-evaluates
			// to collect labels), so it never needs the p_row that embed-all would otherwise produce.
			bool warm_pick = cold && needs_features;
			idx_t warm_leaf = warm_pick ? AIWarmupPickLeaf(row, leaf_values[row], n) : n;
			// Never warm-pick an image leaf that cannot be embedded: it would yield no MLP label and a vision
			// call is expensive. Fall through to the DP instead, which defers the image behind cheaper text
			// leaves via its high cost -- so even while cold a mixed tree evaluates a trainable text leaf first.
			// With the CLIP server an image leaf trains like any other, so it stays a warm-up candidate.
			if (warm_pick &&
			    (warm_leaf >= n || (AITextHasImage(prompt[row][warm_leaf]) && !AIEmbedImagesSupported()))) {
				warm_pick = false;
			}
			if (warm_pick) {
				vector<string> wtexts;
				if (feature_prompt_mode) {
					wtexts.push_back(prompt[row][warm_leaf]);
				} else {
					wtexts.push_back(pred_text[row][warm_leaf]);
					wtexts.push_back(input_text[row][warm_leaf]);
				}
				auto wembs = AIEmbedBatch(wtexts, query_text);
				feats[warm_leaf] = feature_prompt_mode ? wembs[0].embedding
				                                       : BuildPredicateFeature(wembs[0].embedding, wembs[1].embedding);
			}
			// Embed this row's still-undetermined leaves (predicate + input text) in one cache-aware,
			// deduplicated call -- the shared document embeds once -- then JIT-predict p_true against
			// the live model. Embedding happens here (not upfront) so its CPU overlaps the LLM waits.
			if (!warm_pick && needs_features) {
				// With a CLIP image model on the server (ai_embed_images, default) an image leaf is estimable like a
				// text leaf: its predicate text and image embed into one space and the MLP learns the pair. Without
				// one, image embeds are refused anyway, so pay for them only where the estimate can change behavior: a
				// speculative gate on a row that is estimable at all. Plain reorder nodes then keep the neutral p +
				// byte-based cost for image leaves, and a speculative row whose unknown leaves are all images always
				// evaluates regardless.
				const bool images_ok = AIEmbedImagesSupported();
				bool row_estimable = false;
				for (idx_t l = 0; l < n; l++) {
					if (leaf_values[row][l] == AITriState::TRI_UNKNOWN &&
					    (images_ok || !AITextHasImage(prompt[row][l]))) {
						row_estimable = true;
						break;
					}
				}
				const bool embed_images = images_ok || (bind_data.speculative && row_estimable);
				vector<string> texts;
				vector<idx_t> tleaf;
				for (idx_t l = 0; l < n; l++) {
					if (leaf_values[row][l] != AITriState::TRI_UNKNOWN) {
						continue;
					}
					if (AITextHasImage(prompt[row][l])) {
						// Image leaf: embed the predicate TEXT and the image REF through the shared
						// dual-encoder space (mlx-embeddings server). The joint space makes the
						// pred x image cosine a zero-shot selectivity prior; if the embed server
						// cannot do images the embeds fail and the neutral-p fallback below applies.
						const auto open = input_text[row][l].find(AI_IMAGE_OPEN);
						const auto close = input_text[row][l].find(AI_IMAGE_CLOSE);
						if (!embed_images || feature_prompt_mode || open == string::npos || close == string::npos ||
						    close <= open) {
							continue; // no extractable single image ref -> neutral p
						}
						// predicate text marked for the CLIP text tower (the image's space), then the bare
						// sentinel-wrapped ref: the client turns them into image_text / image items
						texts.push_back(string(1, AI_IMAGE_TEXT_MARK) + pred_text[row][l]);
						texts.push_back(input_text[row][l].substr(open, close - open + 1));
						tleaf.push_back(l);
						continue;
					}
					if (feature_prompt_mode) {
						texts.push_back(prompt[row][l]); // single embed(full prompt)
					} else {
						texts.push_back(pred_text[row][l]); // factored: predicate + input
						texts.push_back(input_text[row][l]);
					}
					tleaf.push_back(l);
				}
				// serve prefetched features first; embed only the remainder inline
				vector<string> miss_texts;
				vector<idx_t> miss_leaf;
				{
					std::lock_guard<std::mutex> flk(feat_mutex);
					for (idx_t u = 0; u < tleaf.size(); u++) {
						const idx_t l = tleaf[u];
						auto fit = feat_cache.find(feat_key(row, l));
						if (fit != feat_cache.end()) {
							feats[l] = fit->second;
						} else {
							miss_leaf.push_back(l);
							if (feature_prompt_mode) {
								miss_texts.push_back(texts[u]);
							} else {
								miss_texts.push_back(texts[2 * u]);
								miss_texts.push_back(texts[2 * u + 1]);
							}
						}
					}
				}
				auto embs = AIEmbedBatch(miss_texts, query_text);
				for (idx_t u = 0; u < miss_leaf.size(); u++) {
					const idx_t l = miss_leaf[u];
					if (feature_prompt_mode) {
						feats[l] = embs[u].embedding;
					} else {
						feats[l] = BuildPredicateFeature(embs[2 * u].embedding, embs[2 * u + 1].embedding);
					}
				}
				for (idx_t u = 0; u < tleaf.size(); u++) {
					const idx_t l = tleaf[u];
					if (!feats[l].empty()) {
						p_row[l] = AIEstimateFilterTrueProb(feats[l].data(), feats[l].size());
					}
				}
			}
			// Speculative pre-filter gate: on a row's first scheduling (every leaf still unknown), roll up
			// the node's P(pass) under leaf independence and pass likely-to-pass rows (P >= threshold)
			// through with NO LLM call -- the pulled-up filter above the join re-checks them (a cache
			// hit); only likely-to-fail rows fall through to real evaluation, pruning confirmed-false rows
			// before the join. WARM-UP: while the shared cold-MLP latch is cold (threshold>0), rows ALWAYS
			// evaluate (the gate is skipped) so the MLP collects its first labels before we trust its
			// estimates -- without it a cold MLP that happens to predict "pass" would pass everything
			// through and never learn. threshold==0 -> never warms and P(pass)>=0 always holds, so every
			// row passes through (== the old placeholder); threshold==1 -> almost nothing passes (full
			// pushdown). Result-preserving regardless of the estimate. Off for the plain reorder node.
			if (bind_data.speculative) {
				bool first_schedule = true;
				for (idx_t l = 0; l < n; l++) {
					if (leaf_values[row][l] != AITriState::TRI_UNKNOWN) {
						first_schedule = false;
						break;
					}
				}
				if (first_schedule) {
					// While the MLP is cold (shared latch), evaluate unconditionally to collect labels;
					// once warm, pass the likely-to-pass rows through. A row whose unknown leaves are all
					// image leaves has NO estimate (neutral 0.5 stand-ins only), and this node's purpose is
					// pruning the join input below a fan-out -- passing such rows through on a fictitious
					// estimate un-prunes the join (a 4-way image join explodes combinatorially). Always
					// evaluate them: the call dedups with the pulled-up recheck above, so pruning is the
					// only effect.
					// Image leaves stay excluded here even when they embed (CLIP): a zero-shot image
					// estimate is not calibrated enough to un-prune a join input on; the reorder still
					// uses the feature for ordering, which cannot change a result.
					bool estimable = false;
					for (idx_t l = 0; l < n; l++) {
						if (leaf_values[row][l] == AITriState::TRI_UNKNOWN && !AITextHasImage(prompt[row][l])) {
							estimable = true;
							break;
						}
					}
					const bool warming = cold && bind_data.threshold > 0.0;
					if (estimable && !warming && AIFilterTreeEstimateSelectivity(tree, p_row) >= bind_data.threshold) {
						row_result[row] = 1; // pass-through (true); no LLM, no label
						note_pass(row);
						if (resolved.fetch_add(1) + 1 >= rep_count) {
							std::lock_guard<std::mutex> lock(q_mutex);
							q_cv.notify_all();
						}
						continue;
					}
				}
			}
			vector<double> eff_cost(n);
			for (idx_t l = 0; l < n; l++) {
				eff_cost[l] = cost[row][l] * domain_frac[l];
			}
			idx_t leaf = warm_pick ? warm_leaf : AIFilterTreeChooseNextLeaf(tree, leaf_values[row], p_row, eff_cost);
			if (leaf >= n) {
				// A concurrent waiter routed this row's last unknown leaf between the fold check above and
				// the pick: nothing left to evaluate. Re-fold and resolve; fall back to the first unknown
				// leaf if the pick failed for any other reason.
				const auto refolded = AIFilterTreeEval(tree, leaf_values[row]);
				if (refolded != AITriState::TRI_UNKNOWN) {
					row_result[row] = (refolded == AITriState::TRI_TRUE) ? 1 : 0;
					if (row_result[row]) {
						note_pass(row);
					}
					if (resolved.fetch_add(1) + 1 >= rep_count) {
						std::lock_guard<std::mutex> lock(q_mutex);
						q_cv.notify_all();
					}
					continue;
				}
				for (idx_t l = 0; l < n; l++) {
					if (leaf_values[row][l] == AITriState::TRI_UNKNOWN) {
						leaf = l;
						break;
					}
				}
			}
			// Single-flight this leaf's prompt. Whoever inserts the key first is the fetcher. A worker
			// that finds it already READY applies the result now; one that finds it IN-FLIGHT parks its
			// (row, leaf) and -- when the queue is too thin to keep the pool busy -- PULLS AHEAD to the
			// row's next-best leaf whose prompt is NOT in flight instead of idling (deterministic
			// convergent orders otherwise funnel every worker onto a handful of shared leaf prompts,
			// collapsing effective concurrency; the fork's q10 9,213s regression).
			PromptEval *self = nullptr;
			for (;;) {
				std::unique_lock<std::mutex> lk(pc_mutex);
				auto ins = prompt_cache.emplace(prompt[row][leaf] + leaf_key[leaf], PromptEval {});
				if (ins.second) {
					self = &ins.first->second; // we are the fetcher; pointer stable across rehash
					break;
				}
				PromptEval &pe = ins.first->second;
				if (pe.ready) {
					const bool val = pe.value;
					lk.unlock();
					route_row(row, leaf, val); // result already known: apply now
					break;
				}
				pe.waiters.emplace_back(row, leaf); // in-flight: park; the fetcher routes us
				// Starvation gate: only pull ahead when the queue cannot keep the pool busy.
				bool thin;
				{
					std::lock_guard<std::mutex> qlk(q_mutex);
					thin = workq.size() < nworkers_hint;
				}
				if (!thin) {
					break;
				}
				// Next-best leaf that is UNKNOWN and not already in flight: inflate the cost of every
				// in-flight/known leaf so the DP skips it; bail if nothing evaluable remains.
				vector<double> masked = eff_cost;
				bool any = false;
				for (idx_t l = 0; l < n; l++) {
					if (leaf_values[row][l] != AITriState::TRI_UNKNOWN ||
					    prompt_cache.find(prompt[row][l] + leaf_key[l]) != prompt_cache.end()) {
						masked[l] = 1e30;
					} else {
						any = true;
					}
				}
				if (!any) {
					break;
				}
				const idx_t alt = AIFilterTreeChooseNextLeaf(tree, leaf_values[row], p_row, masked);
				if (alt >= n || masked[alt] >= 1e29 || alt == leaf) {
					break;
				}
				leaf = alt; // loop: try to become this leaf's fetcher (row stays parked on the original)
			}
			if (!self) {
				continue;
			}
			// Evaluate this leaf's AI function on its baked prompt + apply the comparison -> boolean. The
			// rewrite bakes the exact scalar-function prompt, so the call matches (result-preserving); a
			// plain ai_filter leaf (meta kind F) is the original path.
			bool value_ok = false;
			const auto llm_t0 = std::chrono::steady_clock::now();
			const bool value = AIEvalLeaf(context, leaf_meta[leaf], prompt[row][leaf], query_text, value_ok, nullptr, 0,
			                              &pred_text[row][leaf], &batch_config);
			ema_update(llm_lat_us,
			           std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - llm_t0)
			               .count());
			if (value_ok && !feats[leaf].empty()) {
				// The real (function + comparison) outcome is the ground-truth label for the selectivity MLP.
				// Skip leaves with no embedding (image leaves, or a failed embed): a pseudo-random selectivity
				// stands in for the MLP, so there is nothing to train on.
				AIRecordTrainingExample(prompt[row][leaf], feats[leaf], value);
				AISelectivityModel::Global().AddExample(feats[leaf], value);
			}
			vector<std::pair<idx_t, idx_t>> waiters;
			{
				std::lock_guard<std::mutex> lk(pc_mutex);
				self->value = value;
				self->ready = true;
				waiters.swap(self->waiters);
			}
			completions.fetch_add(1);    // only real calls advance the async trainer's cadence
			route_row(row, leaf, value); // the fetcher's own row
			for (auto &w : waiters) {
				route_row(w.first, w.second, value); // rows that parked on this in-flight prompt
			}
		}
	};

	// A worker must never let an exception escape its thread. A wrapper can raise on an answer (a strict cast,
	// say) exactly as the scalar would -- on a call or on a cached answer -- so the first error stops every
	// thread (the same flag a met LIMIT raises) and is rethrown once they have joined.
	auto worker = [&]() {
		try {
			worker_body();
		} catch (std::exception &ex) {
			{
				std::lock_guard<std::mutex> lk(error_mutex);
				if (!first_error.HasError()) {
					first_error = ErrorData(ex);
				}
			}
			{
				std::lock_guard<std::mutex> lock(q_mutex);
				limit_hit.store(true);
			}
			q_cv.notify_all();
		}
	};

	// Dedicated embedding prefetcher: while workers block on in-flight LLM calls, walk the pending
	// queue and pre-embed those rows' unknown text leaves into feat_cache, so the embedding stage
	// rides inside LLM latency instead of adding to it. Purely an accelerator: workers embed any
	// miss inline, so correctness never depends on the prefetcher's progress.
	std::thread prefetcher;
	if (needs_features && !feature_prompt_mode && std::getenv("AISQL_NO_PREFETCH") == nullptr) {
		prefetcher = std::thread([&]() {
			try {
				while (resolved.load() < rep_count && !limit_hit.load()) {
					// snapshot a slice of the queue front
					vector<idx_t> pending;
					idx_t lead_cap = 64;
					if (const char *lc = std::getenv("AISQL_PREFETCH_LEAD")) {
						const int64_t v = atoll(lc);
						lead_cap = v <= 0 ? idx_t(64) : idx_t(v);
					} else {
						const int64_t ll = llm_lat_us.load(std::memory_order_relaxed);
						const int64_t le = embed_lat_us.load(std::memory_order_relaxed);
						if (ll > 0 && le > 0) {
							const idx_t need = idx_t((int64_t(nworkers_hint) * le + ll - 1) / ll);
							lead_cap = MaxValue<idx_t>(nworkers_hint, MinValue<idx_t>(need, 64));
						}
					}
					{
						std::lock_guard<std::mutex> lock(q_mutex);
						for (auto it = workq.begin(); it != workq.end() && pending.size() < lead_cap; ++it) {
							pending.push_back(*it);
						}
					}
					bool worked = false;
					for (const auto r : pending) {
						if (resolved.load() >= rep_count || limit_hit.load()) {
							return;
						}
						// gather this row's unknown, uncached, non-image leaves
						vector<string> texts;
						vector<idx_t> lv;
						{
							std::lock_guard<std::mutex> flk(feat_mutex);
							for (idx_t l = 0; l < n; l++) {
								if (leaf_values[r][l] != AITriState::TRI_UNKNOWN || feat_cache.count(feat_key(r, l))) {
									continue;
								}
								if (AITextHasImage(prompt[r][l])) {
									// image leaf: pre-embed predicate text + the image ref through the
									// dual encoder (the slowest embeds - exactly what overlap is for);
									// same gate as the inline stage: only where the estimate can gate
									bool r_estimable = false;
									for (idx_t l2 = 0; l2 < n; l2++) {
										if (leaf_values[r][l2] == AITriState::TRI_UNKNOWN &&
										    !AITextHasImage(prompt[r][l2])) {
											r_estimable = true;
											break;
										}
									}
									if (!bind_data.speculative || !r_estimable) {
										continue;
									}
									const auto open = input_text[r][l].find(AI_IMAGE_OPEN);
									const auto close = input_text[r][l].find(AI_IMAGE_CLOSE);
									if (open == string::npos || close == string::npos || close <= open) {
										continue;
									}
									texts.push_back(pred_text[r][l]);
									texts.push_back(input_text[r][l].substr(open, close - open + 1));
									lv.push_back(l);
									continue;
								}
								texts.push_back(pred_text[r][l]);
								texts.push_back(input_text[r][l]);
								lv.push_back(l);
							}
						}
						if (lv.empty()) {
							continue;
						}
						const auto emb_t0 = std::chrono::steady_clock::now();
						auto embs = AIEmbedBatch(texts, query_text);
						ema_update(embed_lat_us, std::chrono::duration_cast<std::chrono::microseconds>(
						                             std::chrono::steady_clock::now() - emb_t0)
						                                 .count() /
						                             int64_t(lv.size()));
						std::lock_guard<std::mutex> flk(feat_mutex);
						for (idx_t u = 0; u < lv.size(); u++) {
							feat_cache[feat_key(r, lv[u])] =
							    BuildPredicateFeature(embs[2 * u].embedding, embs[2 * u + 1].embedding);
						}
						worked = true;
					}
					if (!worked) {
						std::this_thread::sleep_for(std::chrono::milliseconds(2));
					}
				}
			} catch (...) {
				return; // an accelerator only: a failed embed stops prefetching, the workers embed inline
			}
		});
	}

	// Dedicated async trainer: overlaps MLP training fully with the workers' LLM calls (no worker is
	// ever stolen for training). It fires one step whenever `batch_size` new labels have arrived, so
	// the model keeps improving the just-in-time predictions of rows scheduled later.
	std::thread trainer;
	if (train_enabled) {
		trainer = std::thread([&]() {
			uint64_t trained_through = 0;
			while (resolved.load() < rep_count && !limit_hit.load()) {
				const uint64_t seen = completions.load();
				if (seen - trained_through >= batch_size) {
					AISelectivityModel::Global().TrainInlineIfReady();
					trained_through = seen;
				} else {
					std::this_thread::sleep_for(std::chrono::milliseconds(1)); // wait for more labels
				}
			}
		});
	}

	const idx_t nworkers = rep_count == 0 ? 0 : (batch_size < rep_count ? batch_size : rep_count);
	vector<std::thread> workers;
	workers.reserve(nworkers);
	for (idx_t w = 0; w < nworkers; w++) {
		workers.emplace_back(worker);
	}
	for (auto &t : workers) {
		t.join();
	}
	if (prefetcher.joinable()) {
		prefetcher.join();
	}
	if (trainer.joinable()) {
		trainer.join();
	}
	if (first_error.HasError()) {
		first_error.Throw();
	}

	out_result.assign(count, 0);
	for (idx_t row = 0; row < count; row++) {
		if (row_valid[row]) {
			out_result[row] = row_result[row_rep[row]]; // duplicates read their representative's result
		}
	}
}

// Scalar ai_function_with_embed / speculative_ai_function_with_embed: read the DataChunk into per-(row,
// leaf) text vectors, evaluate the batch, and write the boolean result vector (NULL leaf -> NULL row).
static void AIFilterWithEmbedFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	auto &bind_data = state.expr.Cast<BoundFunctionExpression>().BindInfo()->Cast<AIFilterWithEmbedBindData>();
	const idx_t n = bind_data.leaf_count;

	// Per-(row, leaf): full prompt (for the LLM call + cost) plus the split predicate/input text. The
	// MLP feature [predicate_emb, input_emb, cos_sim] is built INSIDE the worker pool by embedding the
	// predicate/input on demand -- lazily, overlapped with the LLM calls, and dedup-cached so a
	// document is embedded once across all leaves. A NULL leaf prompt -> NULL row output.
	vector<vector<string>> prompt, pred_text, input_text;
	vector<vector<double>> cost;
	vector<char> row_valid;
	AIFilterAppendArgs(args, n, prompt, pred_text, input_text, cost, row_valid);

	vector<char> out_result;
	AIFilterEvaluateBatch(state.GetContext(), bind_data, count, prompt, pred_text, input_text, cost, row_valid,
	                      state.GetContext().GetCurrentQuery(), out_result);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<bool>(result);
	auto &validity = FlatVector::ValidityMutable(result);
	for (idx_t row = 0; row < count; row++) {
		if (!row_valid[row]) {
			validity.SetInvalid(row);
		} else {
			out[row] = out_result[row] != 0;
		}
	}
}

// Bridge for PhysicalAIRegion (streaming dedup operator above a join). Evaluates the AI call `eval_call`
// over the WHOLE buffered child output `input` at once -- one global, deduplicated, fully-concurrent
// batch instead of per-2048-row-chunk waves -- and appends one result Value per input row to
// `result_out` in serial-scan order (NULL where the row's input was NULL). The call's arg expressions
// are evaluated per buffered chunk (cheap string ops) and accumulated, then handed to the same
// AIFilterEvaluateBatch worker pool the scalar node uses, so short-circuit / DP reorder are preserved.
// Collect the child-output column indices a resolved expression reads (its BoundReferenceExpressions).
static void AICollectRefCols(const Expression &e, std::set<idx_t> &cols) {
	if (e.GetExpressionType() == ExpressionType::BOUND_REF) {
		cols.insert(e.Cast<BoundReferenceExpression>().Index());
	}
	ExpressionIterator::EnumerateChildren(e, [&](const Expression &c) { AICollectRefCols(c, cols); });
}

// The child-output columns the AI call reads -- its dedup key.
vector<idx_t> AIDedupKeyCols(const BoundFunctionExpression &eval_call) {
	std::set<idx_t> key_col_set;
	for (auto &child : eval_call.GetChildren()) {
		AICollectRefCols(*child, key_col_set);
	}
	return vector<idx_t>(key_col_set.begin(), key_col_set.end());
}

//===--------------------------------------------------------------------===//
// Per-leaf seams (see ai_dedup.hpp). Argument layout of a folded node: [0] tree, [1..n] prompt per leaf,
// [1+n..2n] predicate text per leaf, [1+2n..3n] input text per leaf, then the appended key columns.
//===--------------------------------------------------------------------===//
idx_t AILeafCount(const BoundFunctionExpression &eval_call) {
	return eval_call.BindInfo()->Cast<AIFilterWithEmbedBindData>().leaf_count;
}

shared_ptr<AIFilterTreeNode> AILeafTree(const BoundFunctionExpression &eval_call) {
	return eval_call.BindInfo()->Cast<AIFilterWithEmbedBindData>().tree;
}

vector<idx_t> AILeafKeyCols(const BoundFunctionExpression &eval_call, idx_t leaf) {
	const idx_t n = AILeafCount(eval_call);
	auto &children = eval_call.GetChildren();
	std::set<idx_t> cols;
	for (const idx_t pos : {1 + leaf, 1 + n + leaf, 1 + 2 * n + leaf}) {
		AICollectRefCols(*children[pos], cols);
	}
	return vector<idx_t>(cols.begin(), cols.end());
}

void AILeafBuildTexts(ClientContext &context, const BoundFunctionExpression &eval_call, idx_t leaf, DataChunk &rows,
                      AILeafTexts &out) {
	const idx_t n = AILeafCount(eval_call);
	auto &children = eval_call.GetChildren();
	ExpressionExecutor executor(context);
	vector<LogicalType> types;
	for (const idx_t pos : {1 + leaf, 1 + n + leaf, 1 + 2 * n + leaf}) {
		executor.AddExpression(*children[pos]);
		types.push_back(children[pos]->GetReturnType());
	}
	DataChunk texts;
	texts.Initialize(BufferAllocator::Get(context), types);
	executor.Execute(rows, texts);
	for (idx_t i = 0; i < texts.size(); i++) {
		const Value pv = texts.data[0].GetValue(i);
		const Value ptv = texts.data[1].GetValue(i);
		const Value itv = texts.data[2].GetValue(i);
		out.valid.push_back(pv.IsNull() ? 0 : 1);
		out.prompt.push_back(pv.IsNull() ? string() : StringValue::Get(pv));
		out.pred_text.push_back(ptv.IsNull() ? string() : StringValue::Get(ptv));
		out.input_text.push_back(itv.IsNull() ? string() : StringValue::Get(itv));
		const auto &prompt = out.prompt.back();
		out.cost.push_back(
		    pv.IsNull() ? 1.0 : (AITextHasImage(prompt) ? AIImageLeafCost(prompt) : AIEstimatePromptCost(prompt)));
	}
}

void AILeafFeatures(const BoundFunctionExpression &eval_call, const AILeafTexts &texts, const string &query_text,
                    vector<vector<float>> &feats_out) {
	const idx_t count = texts.Size();
	feats_out.assign(count, vector<float>());
	const char *feat_env = std::getenv("AI_MLP_FEATURE");
	const bool feature_prompt_mode = feat_env && string(feat_env) == "prompt";
	vector<string> embed_texts;
	vector<idx_t> estimable;
	for (idx_t i = 0; i < count; i++) {
		if (!texts.valid[i]) {
			continue; // NULL prompt: neutral
		}
		if (AITextHasImage(texts.prompt[i])) {
			// Image leaf: predicate text x the image ref through the dual-encoder space (CLIP server),
			// the same pair feature a text leaf gets. Neutral when images cannot be embedded (text-only
			// server, ai_embed_images=false, the prompt-mode experiment knob) or the ref is not a single
			// sentinel-wrapped image.
			const auto open = texts.input_text[i].find(AI_IMAGE_OPEN);
			const auto close = texts.input_text[i].find(AI_IMAGE_CLOSE);
			if (!AIEmbedImagesSupported() || feature_prompt_mode || open == string::npos || close == string::npos ||
			    close <= open) {
				continue;
			}
			estimable.push_back(i);
			embed_texts.push_back(string(1, AI_IMAGE_TEXT_MARK) + texts.pred_text[i]); // CLIP text tower
			embed_texts.push_back(texts.input_text[i].substr(open, close - open + 1));
			continue;
		}
		estimable.push_back(i);
		if (feature_prompt_mode) {
			embed_texts.push_back(texts.prompt[i]);
		} else {
			embed_texts.push_back(texts.pred_text[i]);
			embed_texts.push_back(texts.input_text[i]);
		}
	}
	if (estimable.empty()) {
		return;
	}
	auto embs = AIEmbedBatch(embed_texts, query_text); // ONE batched request for the whole leaf set
	for (idx_t u = 0; u < estimable.size(); u++) {
		if (feature_prompt_mode) {
			if (embs[u].success) {
				feats_out[estimable[u]] = embs[u].embedding;
			}
		} else if (embs[2 * u].success && embs[2 * u + 1].success) {
			feats_out[estimable[u]] = BuildPredicateFeature(embs[2 * u].embedding, embs[2 * u + 1].embedding);
		}
	}
}

double AILeafPredictFeature(const vector<float> &feat) {
	// Cold model: the per-row batch explores with a seeded pick until the first training step; here the
	// neutral 0.5 lets cost alone order the leaves until the first wave's labels warm the model.
	if (feat.empty() || AISelectivityModel::Global().Stats().train_steps == 0) {
		return 0.5;
	}
	return AIEstimateFilterTrueProb(feat.data(), feat.size());
}

void AILeafPredictFeatures(const vector<const vector<float> *> &feats, vector<double> &p_out) {
	p_out.assign(feats.size(), 0.5);
	if (feats.empty() || AISelectivityModel::Global().Stats().train_steps == 0) {
		return;
	}
	idx_t dim = 0;
	vector<const float *> inputs;
	vector<idx_t> slots;
	for (idx_t i = 0; i < feats.size(); i++) {
		if (!feats[i] || feats[i]->empty()) {
			continue;
		}
		if (dim == 0) {
			dim = feats[i]->size();
		}
		if (feats[i]->size() != dim) {
			continue;
		}
		inputs.push_back(feats[i]->data());
		slots.push_back(i);
	}
	if (inputs.empty()) {
		return;
	}
	vector<double> p(inputs.size());
	// A leaf's features are [predicate embedding | input embedding | cosine] (BuildPredicateFeature) with the
	// SAME predicate embedding on every rep: when the batch shares that half, the first layer's product with
	// it is computed once and only the input half is multiplied per feature (half the arithmetic, the same
	// sums in the same order).
	const idx_t prefix_len = (dim - 1) / 2;
	bool shared = dim >= 3 && dim % 2 == 1;
	for (idx_t k = 1; shared && k < inputs.size(); k++) {
		shared = std::memcmp(inputs[k], inputs[0], prefix_len * sizeof(float)) == 0;
	}
	if (shared) {
		vector<const float *> tails(inputs.size());
		for (idx_t k = 0; k < inputs.size(); k++) {
			tails[k] = inputs[k] + prefix_len;
		}
		AISelectivityModel::Global().PredictBatchShared(inputs[0], prefix_len, tails.data(), dim - prefix_len,
		                                                inputs.size(), p.data());
	} else {
		AISelectivityModel::Global().PredictBatch(inputs.data(), inputs.size(), dim, p.data());
	}
	for (idx_t k = 0; k < slots.size(); k++) {
		p_out[slots[k]] = p[k];
	}
}

void AILeafPredict(const BoundFunctionExpression &eval_call, const AILeafTexts &texts, const string &query_text,
                   vector<double> &p_out) {
	p_out.assign(texts.Size(), 0.5);
	if (AISelectivityModel::Global().Stats().train_steps == 0) {
		return; // cold: no embed needed for a neutral answer
	}
	vector<vector<float>> feats;
	AILeafFeatures(eval_call, texts, query_text, feats);
	for (idx_t i = 0; i < feats.size(); i++) {
		p_out[i] = AILeafPredictFeature(feats[i]);
	}
}

uint64_t AISelectivityTrainSteps() {
	return AISelectivityModel::Global().Stats().train_steps;
}

void AILeafEvaluate(ClientContext &context, const BoundFunctionExpression &eval_call, idx_t leaf,
                    const AILeafTexts &texts, const string &query_text, vector<char> &out_result,
                    vector<char> &out_valid) {
	auto &node = eval_call.BindInfo()->Cast<AIFilterWithEmbedBindData>();
	// The leaf's own meta segment (';'-separated, one per leaf; empty = plain ai_filter everywhere).
	string meta;
	if (!node.meta.empty()) {
		idx_t seg = 0;
		size_t start = 0;
		for (size_t i = 0; i <= node.meta.size(); i++) {
			if (i == node.meta.size() || node.meta[i] == ';') {
				if (seg == leaf) {
					meta = node.meta.substr(start, i - start);
					break;
				}
				seg++;
				start = i + 1;
			}
		}
	}
	// A one-leaf tree over this leaf: the batch evaluator then runs exactly the per-leaf slice of its
	// machinery (worker pool, single-flight, speculative gate, training), with nothing to reorder.
	vector<shared_ptr<Expression>> wrappers;
	if (leaf < node.wrappers.size() && node.wrappers[leaf]) {
		wrappers.push_back(node.wrappers[leaf]);
	}
	AIFilterWithEmbedBindData one(AIFilterTreeParse("L0"), "L0", 1, node.speculative, node.threshold, -1, meta,
	                              std::move(wrappers));
	one.stand_down = node.stand_down;
	const idx_t count = texts.Size();
	vector<vector<string>> prompt(count, vector<string>(1)), pred(count, vector<string>(1)),
	    input(count, vector<string>(1));
	vector<vector<double>> cost(count, vector<double>(1, 1.0));
	vector<char> valid(count, 1);
	for (idx_t i = 0; i < count; i++) {
		prompt[i][0] = texts.prompt[i];
		pred[i][0] = texts.pred_text[i];
		input[i][0] = texts.input_text[i];
		cost[i][0] = texts.cost[i];
		valid[i] = texts.valid[i];
	}
	AIFilterEvaluateBatch(context, one, count, prompt, pred, input, cost, valid, query_text, out_result, -1, nullptr);
	out_valid = valid;
}

struct AILeafUnitEvaluator::Impl {
	vector<AILeafMeta> metas;
	bool speculative = false;
	bool train = true;
	idx_t train_every = 20;
	mutable std::atomic<uint64_t> calls {0};
	//! The owning connection's configuration, taken once (every unit of a region runs for one connection).
	mutable std::once_flag config_once;
	mutable AIConfig config;
	const AIConfig &Config(ClientContext &context) const {
		std::call_once(config_once, [&] { config = AIConfigForContext(context); });
		return config;
	}
};

AILeafUnitEvaluator::AILeafUnitEvaluator(const BoundFunctionExpression &eval_call) : impl(make_uniq<Impl>()) {
	auto &node = eval_call.BindInfo()->Cast<AIFilterWithEmbedBindData>();
	impl->metas = AIParseLeafMeta(node.meta, node.leaf_count, node.wrappers);
	impl->speculative = node.speculative;
	impl->train = std::getenv("DUCKDB_AI_NO_TRAIN") == nullptr;
	// One full-batch gradient step every 3 x ai_concurrency calls (AI_MLP_TRAIN_EVERY overrides). Each step
	// trains on the whole example window, but it also makes every stored prediction stale, so the cadence
	// decides how early the order follows the model. agent_bench Q17 (the one query whose calls depend on it):
	// steps every 20-45 calls -> 171-190 calls; 50-75 -> 156 at 36 s; 100+ -> 190-225 (learns too late).
	const char *every = std::getenv("AI_MLP_TRAIN_EVERY");
	const int64_t every_n = every ? std::atoll(every) : 0;
	impl->train_every =
	    every_n > 0 ? static_cast<idx_t>(every_n) : 3 * MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1);
}

AILeafUnitEvaluator::~AILeafUnitEvaluator() = default;

Value AILeafUnitEvaluator::Evaluate(ClientContext &context, idx_t leaf, const string &prompt, const string &pred_text,
                                    const vector<float> &feature, const string &query_text) const {
	const auto &meta = impl->metas[leaf];
	if (impl->speculative) {
		// A speculative leaf only prunes what is already known false; an unknown input passes through.
		bool v = false;
		return Value::BOOLEAN(
		    !AIEvalLeafCached(context, meta, prompt, query_text, v, &pred_text, &impl->Config(context)) || v);
	}
	bool ok = false;
	const bool value =
	    AIEvalLeaf(context, meta, prompt, query_text, ok, nullptr, 0, &pred_text, &impl->Config(context));
	if (ok && !feature.empty()) {
		AIRecordTrainingExample(prompt, feature, value);
		AISelectivityModel::Global().AddExample(feature, value);
	}
	if (impl->train && (impl->calls.fetch_add(1) + 1) % impl->train_every == 0) {
		AISelectivityModel::Global().TrainInlineIfReady();
	}
	if (!ok) {
		return Value(LogicalType::BOOLEAN); // no answer: NULL, so NOT / IS NULL / OR see an unknown, not a false
	}
	return Value::BOOLEAN(value);
}

// A reorder node (evaluated via the worker pool + count-weighted LIMIT early-stop) vs a plain scalar AI call.
bool AIDedupIsFilterNode(const BoundFunctionExpression &eval_call) {
	const auto fname = eval_call.Function().GetName();
	return fname == "ai_function_with_embed" || fname == "speculative_ai_function_with_embed";
}

// Evaluate one wave of distinct representative rows for an ai_function_with_embed node: materialize each
// leaf's prompt/predicate/input text, then run the shared worker pool once over the whole wave (tree
// short-circuit + DP reorder + single-flight, no per-chunk barriers). remaining_limit (>= 0) stops the pool
// once the PASSING reps' fan-out counts (wave_weights) sum to it -- the count-accurate early-stop; -1 = none.
void AIDedupFireWave(ClientContext &context, const BoundFunctionExpression &eval_call, ColumnDataCollection &wave_reps,
                     const string &query_text, int64_t remaining_limit, const vector<idx_t> &wave_weights,
                     vector<char> &out_result, vector<char> &out_valid) {
	auto &bind_data = eval_call.BindInfo()->Cast<AIFilterWithEmbedBindData>();
	const idx_t n = bind_data.leaf_count;
	ExpressionExecutor executor(context, eval_call.GetChildren());
	vector<LogicalType> arg_types;
	for (auto &child : eval_call.GetChildren()) {
		arg_types.push_back(child->GetReturnType());
	}
	vector<vector<string>> prompt, pred_text, input_text;
	vector<vector<double>> cost;
	out_valid.clear();
	DataChunk rep_chunk, arg_chunk;
	rep_chunk.Initialize(BufferAllocator::Get(context), wave_reps.Types());
	arg_chunk.Initialize(BufferAllocator::Get(context), arg_types);
	ColumnDataScanState scan;
	wave_reps.InitializeScan(scan);
	while (wave_reps.Scan(scan, rep_chunk)) {
		arg_chunk.Reset();
		executor.Execute(rep_chunk, arg_chunk);
		AIFilterAppendArgs(arg_chunk, n, prompt, pred_text, input_text, cost, out_valid);
		rep_chunk.Reset();
	}
	AIFilterEvaluateBatch(context, bind_data, prompt.size(), prompt, pred_text, input_text, cost, out_valid, query_text,
	                      out_result, remaining_limit, &wave_weights);
}

// Evaluate any other scalar AI function (ai_classify / ai_score / ai_complete / plain ai_filter) over the
// distinct representatives by running the function itself on the representative chunks -- each chunk is one
// AIBatchComplete batch (deduped, fixed-concurrency), so a fan-out with <= STANDARD_VECTOR_SIZE distinct
// inputs makes all its calls in a single wave. Reuses the scalar function verbatim (its build + parse).
void AIDedupFireWaveScalar(ClientContext &context, const BoundFunctionExpression &eval_call,
                           ColumnDataCollection &wave_reps, vector<Value> &out_values) {
	ExpressionExecutor executor(context, eval_call);
	const LogicalType &result_type = eval_call.GetReturnType();
	DataChunk rep_chunk;
	rep_chunk.Initialize(BufferAllocator::Get(context), wave_reps.Types());
	ColumnDataScanState scan;
	wave_reps.InitializeScan(scan);
	while (wave_reps.Scan(scan, rep_chunk)) {
		Vector result(result_type);
		executor.ExecuteExpression(rep_chunk, result);
		for (idx_t i = 0; i < rep_chunk.size(); i++) {
			out_values.push_back(result.GetValue(i));
		}
		rep_chunk.Reset();
	}
}

// Factorized (AISQLMapData) node eval: the AI function consumes ONE factorized DataChunk directly -- columns
// [0, ncol) are the distinct rep rows, the trailing column is the BIGINT `__count`. Arg expressions read the
// rep-row columns (the trailing `__count` is not referenced); `__count` becomes the fan-out weight. One chunk,
// no CDC scan loop (the AI currency's cardinality floor keeps a chunk <= STANDARD_VECTOR_SIZE).
void AIDedupFireWaveFactorized(ClientContext &context, const BoundFunctionExpression &eval_call, DataChunk &factorized,
                               const string &query_text, int64_t remaining_limit, vector<char> &out_result,
                               vector<char> &out_valid) {
	auto &bind_data = eval_call.BindInfo()->Cast<AIFilterWithEmbedBindData>();
	const idx_t n = bind_data.leaf_count;
	const idx_t count_col = factorized.ColumnCount() - 1; // trailing `__count`
	const idx_t count = factorized.size();
	vector<idx_t> weights(count);
	for (idx_t j = 0; j < count; j++) {
		weights[j] = static_cast<idx_t>(factorized.data[count_col].GetValue(j).GetValue<int64_t>());
	}
	ExpressionExecutor executor(context, eval_call.GetChildren());
	vector<LogicalType> arg_types;
	for (auto &child : eval_call.GetChildren()) {
		arg_types.push_back(child->GetReturnType());
	}
	DataChunk arg_chunk;
	arg_chunk.Initialize(BufferAllocator::Get(context), arg_types);
	executor.Execute(factorized, arg_chunk);
	vector<vector<string>> prompt, pred_text, input_text;
	vector<vector<double>> cost;
	out_valid.clear();
	AIFilterAppendArgs(arg_chunk, n, prompt, pred_text, input_text, cost, out_valid);
	AIFilterEvaluateBatch(context, bind_data, prompt.size(), prompt, pred_text, input_text, cost, out_valid, query_text,
	                      out_result, remaining_limit, &weights);
}

// Factorized scalar eval: run the scalar AI function over the factorized chunk's rep rows (`__count` ignored).
void AIDedupFireWaveScalarFactorized(ClientContext &context, const BoundFunctionExpression &eval_call,
                                     DataChunk &factorized, vector<Value> &out_values) {
	ExpressionExecutor executor(context, eval_call);
	Vector result(eval_call.GetReturnType());
	executor.ExecuteExpression(factorized, result);
	for (idx_t i = 0; i < factorized.size(); i++) {
		out_values.push_back(result.GetValue(i));
	}
}

void AISetSpeculativeStandDown(Expression &call) {
	if (call.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return;
	}
	auto &fn = call.Cast<BoundFunctionExpression>();
	if (!fn.Function().GetName().StartsWith("speculative_ai_")) {
		return;
	}
	auto &info = fn.BindInfo();
	if (info) {
		info->Cast<AIFilterWithEmbedBindData>().stand_down = true;
	}
}

void AISetFilterLimit(Expression &call, int64_t limit) {
	if (call.GetExpressionType() != ExpressionType::BOUND_FUNCTION) {
		return;
	}
	auto &fn = call.Cast<BoundFunctionExpression>();
	if (fn.Function().GetName() != "ai_function_with_embed" || !fn.BindInfo()) {
		return;
	}
	fn.BindInfoMutable()->Cast<AIFilterWithEmbedBindData>().limit = limit;
}

//===--------------------------------------------------------------------===//
// ai_selectivity_stats() -> one row of the online selectivity model's training counters;
// ai_selectivity_reset() -> BIGINT (buffered examples cleared)
//===--------------------------------------------------------------------===//
struct AISelectivityStatsState : public GlobalTableFunctionState {
	AISelectivityStatsState() : emitted(false) {
	}
	AISelectivityStats stats;
	bool emitted;
};

static unique_ptr<FunctionData> AISelectivityStatsBind(ClientContext &context, TableFunctionBindInput &input,
                                                       vector<LogicalType> &return_types, vector<Identifier> &names) {
	names = {"input_dim", "buffered", "examples_seen", "train_steps", "first_loss", "last_loss", "train_seconds"};
	return_types = {LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT, LogicalType::UBIGINT,
	                LogicalType::DOUBLE,  LogicalType::DOUBLE,  LogicalType::DOUBLE};
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> AISelectivityStatsInit(ClientContext &context,
                                                                   TableFunctionInitInput &input) {
	auto result = make_uniq<AISelectivityStatsState>();
	result->stats = AISelectivityModel::Global().Stats();
	return std::move(result);
}

static void AISelectivityStatsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<AISelectivityStatsState>();
	if (data.emitted) {
		return;
	}
	const auto &s = data.stats;
	output.SetChildCardinality(1);
	output.data[0].SetValue(0, Value::UBIGINT(s.input_dim));
	output.data[1].SetValue(0, Value::UBIGINT(s.buffered));
	output.data[2].SetValue(0, Value::UBIGINT(s.examples_seen));
	output.data[3].SetValue(0, Value::UBIGINT(s.train_steps));
	output.data[4].SetValue(0, Value::DOUBLE(s.first_loss));
	output.data[5].SetValue(0, Value::DOUBLE(s.last_loss));
	output.data[6].SetValue(0, Value::DOUBLE(s.train_seconds));
	data.emitted = true;
}

static void AISelectivityResetFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const int64_t cleared = static_cast<int64_t>(AISelectivityModel::Global().Reset());
	const idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<int64_t>(result);
	for (idx_t i = 0; i < count; i++) {
		out[i] = cleared;
	}
}

//===--------------------------------------------------------------------===//
// ai_reorder_choose_leaf(tree VARCHAR, p FLOAT[], cost FLOAT[]) -> BIGINT (leaf index)
// Thin, deterministic (no LLM) wrapper over the order planner for testing the cost-aware DP.
//===--------------------------------------------------------------------===//
static void AIReorderChooseLeafFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	const idx_t count = args.size();
	auto &tree_col = args.data[0];
	auto &p_col = args.data[1];
	auto &cost_col = args.data[2];
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetDataMutable<int64_t>(result);
	auto &validity = FlatVector::ValidityMutable(result);
	for (idx_t row = 0; row < count; row++) {
		Value tv = tree_col.GetValue(row);
		if (tv.IsNull()) {
			validity.SetInvalid(row);
			continue;
		}
		auto tree = AIFilterTreeParse(StringValue::Get(tv));
		if (!tree) {
			validity.SetInvalid(row);
			continue;
		}
		auto pf = ReadEmbedding(p_col, row);
		auto cf = ReadEmbedding(cost_col, row);
		const idx_t n = tree->LeafCount();
		if (pf.size() < n || cf.size() < n) {
			validity.SetInvalid(row);
			continue;
		}
		vector<double> p(n), c(n);
		for (idx_t i = 0; i < n; i++) {
			p[i] = static_cast<double>(pf[i]);
			c[i] = static_cast<double>(cf[i]);
		}
		vector<AITriState> leaf_values(n, AITriState::TRI_UNKNOWN);
		out[row] = static_cast<int64_t>(AIFilterTreeChooseNextLeaf(*tree, leaf_values, p, c));
	}
}

//===--------------------------------------------------------------------===//
// Registration
//===--------------------------------------------------------------------===//
static ScalarFunction MakeVolatile(ScalarFunction function) {
	function.SetStability(FunctionStability::VOLATILE);
	function.SetFallible();
	return function;
}

void RegisterAIFunctions(ExtensionLoader &loader) {
	const auto varchar = LogicalType::VARCHAR;
	const auto varchar_list = LogicalType::LIST(LogicalType::VARCHAR);

	// varargs=ANY so the call still binds after AIKeyBind appends one history-key column per column-ref in the
	// prompt (e.g. ai_filter('..'||a.x||'..'||b.y) -> ai_filter(prompt, a.x, b.y)); the extra keys are consumed
	// by GetRowKeys. Without it a non-constant prompt fails to bind ("ai_filter(VARCHAR, VARCHAR)").
	loader.RegisterFunction(MakeVolatile(ScalarFunction("ai_filter", {varchar}, LogicalType::BOOLEAN, AIFilterFunction,
	                                                    AIKeyBind, nullptr, nullptr, LogicalType::ANY)));

	ScalarFunctionSet ai_classify("ai_classify");
	// varargs=ANY on the base (input, categories) overload accepts the history-key columns AIKeyBind appends --
	// one per column-ref in the input -- so a MULTI-column input like ai_classify(a.x||b.y, [...]) binds (2
	// keys -> 4 args). ANY is safe (no numeric overloads to steal, and keys can be any column type). A 1-key
	// input still resolves to the exact 3-arg overload below, so the (input, categories, instruction) form is
	// unaffected.
	ai_classify.AddFunction(MakeVolatile(ScalarFunction({varchar, varchar_list}, varchar, AIClassifyFunction, AIKeyBind,
	                                                    nullptr, nullptr, LogicalType::ANY)));
	ai_classify.AddFunction(
	    MakeVolatile(ScalarFunction({varchar, varchar_list, varchar}, varchar, AIClassifyFunction, AIKeyBind)));
	ai_classify.AddFunction(MakeVolatile(ScalarFunction({varchar, LogicalType::ANY}, varchar, AIClassifyStructFunction,
	                                                    AIKeyBind, nullptr, nullptr, LogicalType::ANY)));
	loader.RegisterFunction(ai_classify);

	// varargs=VARCHAR (not ANY) on each overload so the call still binds after AIScoreBind appends the input's
	// history-key column(s) (same mechanism as AIKeyBind). VARCHAR (the input/key type) not ANY: ANY matches
	// the integer lo/hi at cost 0 and greedily steals ai_score(x,'q',1,5) from the {v,v,bigint,bigint} overload
	// (int->varchar is expensive, so the exact BIGINT overload still wins). Keys are VARCHAR since they come
	// from the VARCHAR input's column-refs.
	ScalarFunctionSet ai_score("ai_score");
	// 1-arg overload keeps NO varargs: with varargs it makes ai_score(x,'criteria') ambiguous between {v,v} and
	// {v}+vararg. (ai_score(x) with a bare column input + no criteria is not a supported form.)
	ai_score.AddFunction(MakeVolatile(ScalarFunction({varchar}, LogicalType::DOUBLE, AIScoreFunction, AIScoreBind)));
	ai_score.AddFunction(MakeVolatile(ScalarFunction({varchar, varchar}, LogicalType::DOUBLE, AIScoreFunction,
	                                                 AIScoreBind, nullptr, nullptr, varchar)));
	// ai_score(input, criteria, lo, hi): integer bounds -> BIGINT in [lo,hi] (rounded); float bounds -> DOUBLE.
	// Overload resolution picks BIGINT for integer literals, DOUBLE otherwise (mixed widens to DOUBLE).
	ai_score.AddFunction(
	    MakeVolatile(ScalarFunction({varchar, varchar, LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::BIGINT,
	                                AIScoreFunction, AIScoreBind, nullptr, nullptr, varchar)));
	ai_score.AddFunction(
	    MakeVolatile(ScalarFunction({varchar, varchar, LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                                AIScoreFunction, AIScoreBind, nullptr, nullptr, varchar)));
	loader.RegisterFunction(ai_score);

	loader.RegisterFunction(MakeVolatile(ScalarFunction("ai_agg", {varchar_list, varchar}, varchar, AIAggFunction)));

	// Embeddings: no history, fixed path, local sentence-transformers by default.
	loader.RegisterFunction(
	    MakeVolatile(ScalarFunction("ai_embed", {varchar}, LogicalType::LIST(LogicalType::FLOAT), AIEmbedFunction)));

	// ai_image(path_or_url): mark a string as an image for the multimodal AI functions (pure transform).
	loader.RegisterFunction(ScalarFunction("ai_image", {varchar}, varchar, AIImageFunction));

	// Speculative pre-filter: placeholder left where an ai_filter was pulled up from (see optimizer).
	loader.RegisterFunction(MakeVolatile(
	    ScalarFunction("speculative_ai_filter", {varchar}, LogicalType::BOOLEAN, AISpeculativeFilterFunction)));

	ScalarFunctionSet ai_complete("ai_complete");
	// varargs=ANY accepts the hidden history-key args AIKeyBind appends (one per column-ref in the prompt), so
	// a MULTI-column prompt like ai_complete('p '||a.x||b.y) binds. AIPromptFunction tells the optional
	// json_schema arg apart from keys via AIKeyBindData.key_start (declared-arg count), never by position.
	ai_complete.AddFunction(MakeVolatile(
	    ScalarFunction({varchar}, varchar, AIPromptFunction, AIKeyBind, nullptr, nullptr, LogicalType::ANY)));
	ai_complete.AddFunction(MakeVolatile(
	    ScalarFunction({varchar, varchar}, varchar, AIPromptFunction, AIKeyBind, nullptr, nullptr, LogicalType::ANY)));
	loader.RegisterFunction(ai_complete);

	// Usage / cost tracking
	loader.RegisterFunction(TableFunction("ai_usage", {}, AIUsageFunction, AIUsageBind, AIUsageInit));
	loader.RegisterFunction(
	    MakeVolatile(ScalarFunction("ai_usage_reset", {}, LogicalType::BIGINT, AIUsageResetFunction)));
	loader.RegisterFunction(MakeVolatile(ScalarFunction(
	    "ai_local_cache_clear", {}, LogicalType::BIGINT, [](DataChunk &args, ExpressionState &, Vector &result) {
		    result.Reference(Value::BIGINT(static_cast<int64_t>(AILocalCacheClear())), count_t(args.size()));
	    })));

	// History mode
	loader.RegisterFunction(
	    MakeVolatile(ScalarFunction("ai_history_reset", {}, LogicalType::BIGINT, AIHistoryResetFunction)));
	loader.RegisterFunction(MakeVolatile(
	    ScalarFunction("ai_history_mode", {LogicalType::BOOLEAN}, LogicalType::BOOLEAN, AIHistoryModeFunction)));

	// Turbo mode: adaptive (AIMD) concurrency
	loader.RegisterFunction(
	    MakeVolatile(ScalarFunction("ai_turbo", {LogicalType::BOOLEAN}, LogicalType::BOOLEAN, AITurboModeFunction)));

	// Per-row adaptive boolean predicate over ai_filter leaves (inserted by the DUCKDB_AI_REORDER
	// rewrite; also directly callable for testing). It embeds internally and orders per row. varargs
	// (all VARCHAR): the n full prompts, then n predicate texts, then n input texts; arg 0 = tree.
	loader.RegisterFunction(MakeVolatile(ScalarFunction("ai_function_with_embed", {varchar}, LogicalType::BOOLEAN,
	                                                    AIFilterWithEmbedFunction, AIFilterWithEmbedBind, nullptr,
	                                                    nullptr, LogicalType::ANY, FunctionStability::VOLATILE)));

	// Speculative variant: same node + arg contract, but row-adaptive partial pushdown -- pre-evaluates
	// only likely-to-fail rows (MLP-gated) to prune before a join; likely-to-pass rows pass through for
	// the pulled-up filter to re-check. Emitted at the pushed-down position by the semantic pull-up.
	loader.RegisterFunction(MakeVolatile(ScalarFunction(
	    "speculative_ai_function_with_embed", {varchar}, LogicalType::BOOLEAN, AIFilterWithEmbedFunction,
	    AISpeculativeFilterWithEmbedBind, nullptr, nullptr, LogicalType::ANY, FunctionStability::VOLATILE)));

	// Self-labeling training data for the selectivity MLP
	loader.RegisterFunction(
	    TableFunction("ai_filter_training_data", {}, AITrainingFunction, AITrainingBind, AITrainingInit));
	loader.RegisterFunction(
	    MakeVolatile(ScalarFunction("ai_training_reset", {}, LogicalType::BIGINT, AITrainingResetFunction)));

	// Online selectivity model: training counters + reset
	loader.RegisterFunction(TableFunction("ai_selectivity_stats", {}, AISelectivityStatsFunction,
	                                      AISelectivityStatsBind, AISelectivityStatsInit));
	loader.RegisterFunction(
	    MakeVolatile(ScalarFunction("ai_selectivity_reset", {}, LogicalType::BIGINT, AISelectivityResetFunction)));

	// Deterministic (no-LLM) hook over the cost-aware order planner, for testing.
	loader.RegisterFunction(MakeVolatile(ScalarFunction(
	    "ai_reorder_choose_leaf",
	    {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::LIST(LogicalType::FLOAT)},
	    LogicalType::BIGINT, AIReorderChooseLeafFunction)));
}

//===--------------------------------------------------------------------===//
// Factor-graph decomposition of a conjunctive folded node
//===--------------------------------------------------------------------===//

void AIFactorPromptOperands(const Expression &call_prompt, vector<const Expression *> &ops) {
	if (call_prompt.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
		auto &fn = call_prompt.Cast<BoundFunctionExpression>();
		if (fn.Function().GetName() == "concat" || fn.Function().GetName() == "||") {
			for (auto &child : fn.GetChildren()) {
				AIFactorPromptOperands(*child, ops);
			}
			return;
		}
	}
	ops.push_back(&call_prompt);
}

//! Text of a constant expression, or of a concat chain of constants (what AISplitPrompt's
//! BuildConcat yields for a leaf's predicate part when the optimizer has not folded it). False
//! for anything row-varying.
static bool AIConstantConcatText(const Expression &expr, string &out) {
	if (expr.GetExpressionType() == ExpressionType::VALUE_CONSTANT) {
		const Value &v = expr.Cast<BoundConstantExpression>().GetValue();
		if (!v.IsNull()) {
			out += v.ToString();
		}
		return true;
	}
	if (expr.GetExpressionType() == ExpressionType::BOUND_FUNCTION) {
		auto &fn = expr.Cast<BoundFunctionExpression>();
		const auto name = fn.Function().GetName();
		if (name == "concat" || name == "||") {
			for (auto &child : fn.GetChildren()) {
				if (!AIConstantConcatText(*child, out)) {
					return false;
				}
			}
			return true;
		}
	}
	return false;
}

bool AIFactorEvalUnit(ClientContext &context, const BoundFunctionExpression &sub_node,
                      const vector<string> &leaf_prompts, const string &query_text, bool &valid,
                      const vector<string> *leaf_prefixes, idx_t expected_reuse) {
	const AIConfig unit_config = AIConfigForContext(context);
	auto &bind_data = sub_node.BindInfo()->Cast<AIFilterWithEmbedBindData>();
	const auto metas = AIParseLeafMeta(bind_data.meta, bind_data.leaf_count, bind_data.wrappers);
	const auto &children = sub_node.GetChildren();
	valid = true;
	for (idx_t l = 0; l < bind_data.leaf_count; l++) {
		bool ok = false;
		const string *prefix = leaf_prefixes && l < leaf_prefixes->size() ? &(*leaf_prefixes)[l] : nullptr;
		// The leaf's predicate text is the node's constant feature column [1 + n + l].
		string pred;
		const idx_t pred_col = 1 + bind_data.leaf_count + l;
		if (pred_col >= children.size() || !AIConstantConcatText(*children[pred_col], pred)) {
			pred.clear();
		}
		// A wrapper that raises on the answer propagates: the graph's pool hands the error to its driver.
		const bool value =
		    AIEvalLeaf(context, metas[l], leaf_prompts[l], query_text, ok, prefix, expected_reuse, &pred, &unit_config);
		if (!ok) {
			valid = false;
			return false;
		}
		if (!value) {
			return false; // conjunctive sub-node: first FALSE decides
		}
	}
	return true;
}

static void AIFactorCollectRefs(const Expression &expr, vector<idx_t> &out) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		out.push_back(expr.Cast<BoundReferenceExpression>().Index());
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) { AIFactorCollectRefs(child, out); });
}

bool AIFactorDecompose(const BoundFunctionExpression &node, const vector<idx_t> &side_of,
                       vector<AIFactorLeaf> &out_leaves) {
	if (!node.BindInfo()) {
		return false;
	}
	auto &bind_data = node.BindInfo()->Cast<AIFilterWithEmbedBindData>();
	auto &tree = *bind_data.tree;
	// pure conjunction only: a single leaf, or AND over leaves (per-leaf falsity must kill the row)
	if (tree.type == AIFilterTreeType::AND_OP) {
		for (auto &child : tree.children) {
			if (child->type != AIFilterTreeType::LEAF) {
				return false;
			}
		}
	} else if (tree.type != AIFilterTreeType::LEAF) {
		return false;
	}
	const idx_t n = bind_data.leaf_count;
	out_leaves.assign(n, {});
	for (idx_t l = 0; l < n; l++) {
		vector<idx_t> refs;
		for (idx_t part = 0; part < 3; part++) {
			AIFactorCollectRefs(*node.GetChildren()[1 + part * n + l], refs);
		}
		auto &sides = out_leaves[l].sides;
		for (const auto col : refs) {
			if (col >= side_of.size()) {
				return false;
			}
			if (std::find(sides.begin(), sides.end(), side_of[col]) == sides.end()) {
				sides.push_back(side_of[col]);
			}
		}
		std::sort(sides.begin(), sides.end());
		if (sides.empty() || sides.size() > 2) {
			return false;
		}
	}
	return true;
}

static void AIFactorRemapRefs(Expression &expr, const vector<idx_t> &index_map) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		auto &ref = expr.Cast<BoundReferenceExpression>();
		if (ref.Index() >= index_map.size() || index_map[ref.Index()] == DConstants::INVALID_INDEX) {
			throw InternalException("AIFactorSubNode: leaf references a column outside its sides");
		}
		ref.IndexMutable() = index_map[ref.Index()];
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { AIFactorRemapRefs(child, index_map); });
}

unique_ptr<Expression> AIFactorSubNode(const BoundFunctionExpression &node, const vector<idx_t> &leaf_ids,
                                       const vector<idx_t> &index_map) {
	auto &bind_data = node.BindInfo()->Cast<AIFilterWithEmbedBindData>();
	const idx_t n = bind_data.leaf_count;
	const idx_t m = leaf_ids.size();
	string tree_str;
	if (m == 1) {
		tree_str = "L0";
	} else {
		tree_str = "A(";
		for (idx_t i = 0; i < m; i++) {
			tree_str += (i ? ",L" : "L") + std::to_string(i);
		}
		tree_str += ")";
	}
	// meta subset (';'-joined per-leaf tokens; empty = all plain ai_filter)
	string meta;
	if (!bind_data.meta.empty()) {
		vector<string> tokens;
		string cur;
		for (const char c : bind_data.meta) {
			if (c == ';') {
				tokens.push_back(cur);
				cur.clear();
			} else {
				cur += c;
			}
		}
		tokens.push_back(cur);
		for (idx_t i = 0; i < m; i++) {
			if (i) {
				meta += ';';
			}
			meta += leaf_ids[i] < tokens.size() ? tokens[leaf_ids[i]] : "F";
		}
	}
	vector<unique_ptr<Expression>> args;
	args.push_back(make_uniq<BoundConstantExpression>(Value(tree_str)));
	for (idx_t part = 0; part < 3; part++) {
		for (idx_t i = 0; i < m; i++) {
			auto arg = node.GetChildren()[1 + part * n + leaf_ids[i]]->Copy();
			AIFactorRemapRefs(*arg, index_map);
			args.push_back(std::move(arg));
		}
	}
	if (!meta.empty()) {
		args.push_back(make_uniq<BoundConstantExpression>(Value(meta)));
	}
	vector<shared_ptr<Expression>> wrappers;
	if (!bind_data.wrappers.empty()) {
		wrappers.resize(m);
		for (idx_t i = 0; i < m; i++) {
			wrappers[i] = leaf_ids[i] < bind_data.wrappers.size() ? bind_data.wrappers[leaf_ids[i]] : nullptr;
		}
	}
	shared_ptr<AIFilterTreeNode> tree = AIFilterTreeParse(tree_str);
	auto sub_bind = make_uniq<AIFilterWithEmbedBindData>(std::move(tree), tree_str, m, /*speculative=*/false,
	                                                     /*threshold=*/0.5, /*limit=*/-1, meta, std::move(wrappers));
	return make_uniq<BoundFunctionExpression>(node.Function(), std::move(args), std::move(sub_bind));
}

} // namespace duckdb
