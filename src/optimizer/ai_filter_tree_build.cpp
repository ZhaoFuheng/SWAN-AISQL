#include "optimizer/ai_filter_tree_build.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/types/blob.hpp"

#include <cmath>

namespace duckdb {

// Format a score bound for the prompt/meta: integers without decimals, doubles with trailing zeros stripped.
// Kept BYTE-IDENTICAL with AIFormatBound in ai_functions.cpp so the reorder bakes ai_score's exact prompt.
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

bool AIIsFilterLeaf(const Expression &expr) {
	return expr.GetExpressionType() == ExpressionType::BOUND_FUNCTION &&
	       expr.Cast<BoundFunctionExpression>().Function().GetName() == "ai_filter";
}

bool AIIsPureFilterBoolean(const Expression &expr) {
	if (AIIsFilterLeaf(expr)) {
		return true;
	}
	const auto type = expr.GetExpressionType();
	if (type == ExpressionType::CONJUNCTION_AND || type == ExpressionType::CONJUNCTION_OR) {
		auto &conj = expr.Cast<BoundConjunctionExpression>();
		if (conj.GetChildren().empty()) {
			return false;
		}
		for (auto &child : conj.GetChildren()) {
			if (!AIIsPureFilterBoolean(*child)) {
				return false;
			}
		}
		return true;
	}
	if (type == ExpressionType::OPERATOR_NOT) {
		auto &op = expr.Cast<BoundOperatorExpression>();
		return op.GetChildren().size() == 1 && AIIsPureFilterBoolean(*op.GetChildren()[0]);
	}
	return false;
}

idx_t AICountFilterLeaves(const Expression &expr) {
	if (AIIsFilterLeaf(expr)) {
		return 1;
	}
	idx_t n = 0;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) { n += AICountFilterLeaves(child); });
	return n;
}

//! Normalize one predicate's subtree. Deliberately does NOT renumber leaves: callers build
//! SEVERAL predicates into ONE shared `leaves` vector and AND the subtrees together, so indices
//! are positions in that shared vector, not 0..n-1 for this subtree. Compacting here renumbered
//! the leaves of predicates built earlier and silently dropped them (`a AND b`, split by DuckDB
//! into two filter expressions, became `A(L0,L0)` over b alone). Absorption can leave a leaf
//! unreferenced; that costs an unused argument slot -- a string concat, never an LLM call, since
//! only leaves reachable in the tree are ever evaluated.
static unique_ptr<AIFilterTreeNode> NormalizeSubtree(unique_ptr<AIFilterTreeNode> tree) {
	return AINormalizeFilterTree(std::move(tree));
}

//! The same prompt is the same predicate: reuse its leaf rather than adding a second one. Without
//! this, `ai_filter(p) AND (ai_filter(p) OR ai_filter(q))` carries two independent leaves for p,
//! which the DP orderer then costs as if they could disagree.
static unique_ptr<AIFilterTreeNode> BuildFilterTreeRec(const Expression &expr,
                                                       vector<unique_ptr<Expression>> &leaf_prompts) {
	if (AIIsFilterLeaf(expr)) {
		auto &prompt = *expr.Cast<BoundFunctionExpression>().GetChildren()[0];
		for (idx_t i = 0; i < leaf_prompts.size(); i++) {
			if (leaf_prompts[i]->Equals(prompt)) {
				return AIFilterTreeNode::Leaf(i);
			}
		}
		const idx_t idx = leaf_prompts.size();
		leaf_prompts.push_back(prompt.Copy());
		return AIFilterTreeNode::Leaf(idx);
	}
	const auto type = expr.GetExpressionType();
	if (type == ExpressionType::CONJUNCTION_AND || type == ExpressionType::CONJUNCTION_OR) {
		auto &conj = expr.Cast<BoundConjunctionExpression>();
		const auto tree_type =
		    type == ExpressionType::CONJUNCTION_AND ? AIFilterTreeType::AND_OP : AIFilterTreeType::OR_OP;
		// Build an n-ary node: splice a same-type child's children directly in rather than nesting,
		// so `A AND (B AND C)` becomes a flat 3-way AND regardless of how the input was associated.
		vector<unique_ptr<AIFilterTreeNode>> children;
		for (auto &child : conj.GetChildren()) {
			auto sub = BuildFilterTreeRec(*child, leaf_prompts);
			if (sub->type == tree_type) {
				for (auto &grandchild : sub->children) {
					children.push_back(std::move(grandchild));
				}
			} else {
				children.push_back(std::move(sub));
			}
		}
		return AIFilterTreeNode::Op(tree_type, std::move(children));
	}
	// OPERATOR_NOT
	auto &op = expr.Cast<BoundOperatorExpression>();
	vector<unique_ptr<AIFilterTreeNode>> children;
	children.push_back(BuildFilterTreeRec(*op.GetChildren()[0], leaf_prompts));
	return AIFilterTreeNode::Op(AIFilterTreeType::NOT_OP, std::move(children));
}

unique_ptr<AIFilterTreeNode> AIBuildFilterTree(const Expression &expr, vector<unique_ptr<Expression>> &leaf_prompts) {
	return NormalizeSubtree(BuildFilterTreeRec(expr, leaf_prompts));
}

static unique_ptr<Expression> BuildConcat(ClientContext &context, vector<unique_ptr<Expression>> parts) {
	if (parts.size() == 1) {
		return std::move(parts[0]);
	}
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto &entry = catalog.GetEntry<ScalarFunctionCatalogEntry>(
	    context, QualifiedName(catalog.GetName(), Identifier::DefaultSchema(), "concat"));
	FunctionBinder function_binder(context);
	ErrorData error;
	return function_binder.BindScalarFunction(entry, std::move(parts), error);
}

//! Recursively flatten a `||`/concat chain into its leaf operands (so `a || b || c` and any nesting
//! yield [a, b, c]); a non-concat expression is itself a single operand.
static void FlattenConcat(const Expression &expr, vector<const Expression *> &out) {
	if (expr.GetExpressionType() == ExpressionType::BOUND_FUNCTION) {
		auto &fn = expr.Cast<BoundFunctionExpression>();
		const auto name = fn.Function().GetName();
		if (name == "concat" || name == "||") {
			for (auto &child : fn.GetChildren()) {
				FlattenConcat(*child, out);
			}
			return;
		}
	}
	out.push_back(&expr);
}

//! Look past the all-columns-NULL guard AIKeyBind wraps a prompt in
//! (`CASE WHEN c IS NULL ... THEN NULL ELSE <prompt> END`). It is a wrapper around the prompt, not
//! part of it, and every structural analysis -- the predicate/input split, and through it the
//! prefix-cache prefix -- must see the concatenation underneath. Missing this silently disabled
//! explicit prompt caching (prefix_cache.test: 35 breakpoint requests -> 0).
const Expression &AIUnwrapNullGuard(const Expression &expr) {
	const Expression *result = &expr; // the argument always outlives the call: it is a plan node's expression
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_CASE) {
		return *result;
	}
	auto &case_expr = expr.Cast<BoundCaseExpression>();
	if (case_expr.CaseChecks().size() != 1) {
		return *result;
	}
	auto &then_expr = *case_expr.CaseChecks()[0].then_expr;
	if (then_expr.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT ||
	    !then_expr.Cast<BoundConstantExpression>().GetValue().IsNull()) {
		return *result; // someone else's CASE: leave it alone
	}
	return case_expr.Else();
}

std::pair<unique_ptr<Expression>, unique_ptr<Expression>> AISplitPrompt(ClientContext &context,
                                                                        const Expression &guarded_prompt) {
	const Expression &prompt = AIUnwrapNullGuard(guarded_prompt);
	// A `question || document` prompt splits into constant (question) and row-varying (document)
	// parts. Embedding them separately lets the MLP see [predicate, input, cos_sim] and lets the
	// shared document dedup in the embed cache to one encoding per row instead of one per leaf.
	// Flattening handles multi-input prompts too, e.g. `'is ' || a || ' same as ' || b` -> predicate
	// = the constant text, input = a || b (all row-varying parts lumped into one input embedding).
	// v1 lumps multiple docs into one `emb(a || b)` (Option A). Future work: embed each doc
	// separately and pool + add a doc<->doc cosine feature (better for comparison filters).
	vector<const Expression *> operands;
	FlattenConcat(prompt, operands);
	if (operands.size() >= 2) {
		vector<unique_ptr<Expression>> consts, vars;
		for (auto *op : operands) {
			if (op->IsFoldable()) {
				consts.push_back(op->Copy());
			} else {
				vars.push_back(op->Copy());
			}
		}
		if (!consts.empty() && !vars.empty()) {
			auto pred = BuildConcat(context, std::move(consts));
			auto input = BuildConcat(context, std::move(vars));
			if (pred && input) {
				return {std::move(pred), std::move(input)};
			}
		}
	}
	// No clean split: predicate == input (cos_sim will be 1); still correct, just no factorization.
	return {prompt.Copy(), prompt.Copy()};
}

//===----------------------------------------------------------------------===//
// Mixed AI-comparison leaves (ai_filter / ai_classify=x / ai_score>v / ai_complete=x)
//===----------------------------------------------------------------------===//

// Meta kind char for an AI call usable as a comparison leaf, else 0.
static char AIComparableKind(const Expression &e) {
	if (e.GetExpressionType() != ExpressionType::BOUND_FUNCTION) {
		return 0;
	}
	const auto &name = e.Cast<BoundFunctionExpression>().Function().GetName();
	if (name == "ai_classify") {
		return 'C';
	}
	if (name == "ai_score") {
		return 'S';
	}
	if (name == "ai_complete") {
		return 'M';
	}
	return 0;
}

// AIKeyBind appends one hidden history-key argument per DISTINCT column-ref in the input (arg 0), so the
// number of DECLARED args = total children - (distinct col-refs in the input). Recomputing it here matches
// AIKeyBindData::key_start exactly, without needing that private bind data, and lets us separate declared
// args from the history keys.
//
// DISTINCT matters since the prompt carries a NULL guard: `CASE WHEN c IS NULL THEN NULL ELSE ... c ... END`
// mentions every column twice. Counting occurrences instead of columns made key_start too small, every
// leaf-kind check below rejected, and the reorder silently stopped folding ANY prompt built from columns.
static void AICollectDistinctColRefs(const Expression &expr, vector<const Expression *> &out) {
	if (expr.GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
		for (auto *seen : out) {
			if (seen->Equals(expr)) {
				return;
			}
		}
		out.push_back(&expr);
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) { AICollectDistinctColRefs(child, out); });
}

static idx_t AICountColRefs(const Expression &expr) {
	vector<const Expression *> distinct;
	AICollectDistinctColRefs(expr, distinct);
	return distinct.size();
}

// Comparison char; flips direction when the AI call is the right-hand operand.
static char AICmpOp(ExpressionType t, bool flipped) {
	switch (t) {
	case ExpressionType::COMPARE_EQUAL:
		return 'e';
	case ExpressionType::COMPARE_NOTEQUAL:
		return 'n';
	case ExpressionType::COMPARE_GREATERTHAN:
		return flipped ? 'l' : 'g';
	case ExpressionType::COMPARE_LESSTHAN:
		return flipped ? 'g' : 'l';
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return flipped ? 'L' : 'G';
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return flipped ? 'G' : 'L';
	default:
		return 0;
	}
}

static const char *AIOpName(char op) {
	switch (op) {
	case 'n':
		return "ne";
	case 'g':
		return "gt";
	case 'l':
		return "lt";
	case 'G':
		return "ge";
	case 'L':
		return "le";
	case 'i':
		return "in";
	case 'I':
		return "ni";
	default:
		return "eq";
	}
}

static const char *AIOpPhrase(char op) {
	switch (op) {
	case 'n':
		return "not";
	case 'g':
		return "above";
	case 'l':
		return "below";
	case 'G':
		return "at least";
	case 'L':
		return "at most";
	case 'i':
		return "one of";
	case 'I':
		return "not one of";
	default:
		return "exactly";
	}
}

// If `expr` is `OR(=(aicall,c1), =(aicall,c2), ...)` where every equality tests the SAME
// ai_classify/ai_complete call against a foldable constant (the desugared form of `ai_classify(x) IN
// (...)`), fill the shared call + kind + the constant set and return true. Classify/complete only (string
// results); >=2 members.
static bool AIDetectSameCallEqualities(const Expression &expr, const Expression *&out_ai, char &out_kind,
                                       vector<const Expression *> &out_csts) {
	if (expr.GetExpressionType() != ExpressionType::CONJUNCTION_OR) {
		return false;
	}
	auto &conj = expr.Cast<BoundConjunctionExpression>();
	if (conj.GetChildren().size() < 2) {
		return false;
	}
	const Expression *ai0 = nullptr;
	char kind0 = 0;
	vector<const Expression *> csts;
	for (auto &child : conj.GetChildren()) {
		if (!BoundComparisonExpression::IsComparison(*child) ||
		    child->GetExpressionType() != ExpressionType::COMPARE_EQUAL) {
			return false;
		}
		auto &cmp = child->Cast<BoundFunctionExpression>();
		const Expression &lhs = BoundComparisonExpression::Left(cmp);
		const Expression &rhs = BoundComparisonExpression::Right(cmp);
		const char lk = AIComparableKind(lhs);
		const char rk = AIComparableKind(rhs);
		const Expression *ai = nullptr;
		const Expression *cst = nullptr;
		char kind = 0;
		if ((lk == 'C' || lk == 'M') && rhs.IsFoldable()) {
			ai = &lhs;
			cst = &rhs;
			kind = lk;
		} else if ((rk == 'C' || rk == 'M') && lhs.IsFoldable()) {
			ai = &rhs;
			cst = &lhs;
			kind = rk;
		} else {
			return false;
		}
		if (!ai0) {
			ai0 = ai;
			kind0 = kind;
		} else if (kind != kind0 || !Expression::Equals(*ai0, *ai)) {
			return false; // must be the SAME ai call
		}
		csts.push_back(cst);
	}
	out_ai = ai0;
	out_kind = kind0;
	out_csts = std::move(csts);
	return true;
}

//===----------------------------------------------------------------------===//
// Wrapped leaves: any boolean expression whose only row-varying input is one ai_* call
//===----------------------------------------------------------------------===//

// Meta kind char of an AI call (F for ai_filter, else the comparable kind), or 0 for anything else.
static char AILeafCallKind(const Expression &e) {
	return AIIsFilterLeaf(e) ? 'F' : AIComparableKind(e);
}

// Collect the AI calls in `expr` without descending into them (a call's prompt is opaque here).
static void AICollectAICalls(const Expression &expr, vector<const Expression *> &out) {
	if (AILeafCallKind(expr)) {
		out.push_back(&expr);
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) { AICollectAICalls(child, out); });
}

// Everything around the call must be row-constant and cheap: constants, non-volatile functions (comparisons
// and casts are functions), operators, conjunctions and CASE. A column reference, subquery, parameter,
// aggregate, window or lambda disqualifies the expression -- its value is not in the leaf's argument layout.
static bool AIWrapperIsCheap(const Expression &expr) {
	if (AILeafCallKind(expr)) {
		return true;
	}
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_CONSTANT:
	case ExpressionClass::BOUND_OPERATOR:
	case ExpressionClass::BOUND_CONJUNCTION:
	case ExpressionClass::BOUND_CASE:
		break;
	case ExpressionClass::BOUND_FUNCTION:
		if (expr.Cast<BoundFunctionExpression>().Function().GetStability() == FunctionStability::VOLATILE) {
			return false;
		}
		break;
	default:
		return false;
	}
	bool cheap = true;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (cheap && !AIWrapperIsCheap(child)) {
			cheap = false;
		}
	});
	return cheap;
}

// Replace every occurrence of `call` by a reference to column 0 of the call's type (the answer).
static void AIReplaceAICall(unique_ptr<Expression> &expr, const Expression &call) {
	if (Expression::Equals(*expr, call)) {
		expr = make_uniq<BoundReferenceExpression>(Identifier("result"), call.GetReturnType(), 0);
		return;
	}
	ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) { AIReplaceAICall(child, call); });
}

// `expr` is a wrapped leaf when it is boolean, every AI call inside it is the SAME call, and the rest is
// cheap. One call asked once, then the wrapper decides the leaf -- `lower(ai_complete(x)) IN ('yes','true')`
// desugars to an OR of two comparisons and would otherwise be two leaves (or none). A boolean tree over
// ai_filter alone stays a tree of plain leaves, so existing plan shapes are unchanged.
static bool AIDetectWrappedLeaf(const Expression &expr, AIMixedLeaf &out) {
	if (expr.GetReturnType().id() != LogicalTypeId::BOOLEAN) {
		return false;
	}
	vector<const Expression *> calls;
	AICollectAICalls(expr, calls);
	if (calls.empty() || calls[0] == &expr) {
		return false;
	}
	for (auto *call : calls) {
		if (!Expression::Equals(*calls[0], *call)) {
			return false;
		}
	}
	const char kind = AILeafCallKind(*calls[0]);
	const auto type = expr.GetExpressionType();
	const bool tree_node = type == ExpressionType::CONJUNCTION_AND || type == ExpressionType::CONJUNCTION_OR ||
	                       type == ExpressionType::OPERATOR_NOT;
	if (kind == 'F' && tree_node) {
		return false;
	}
	if (!AIWrapperIsCheap(expr)) {
		return false;
	}
	auto wrapper = expr.Copy();
	AIReplaceAICall(wrapper, *calls[0]);
	out = AIMixedLeaf {kind, 'w', calls[0], nullptr, {}};
	out.wrapper = std::move(wrapper);
	return true;
}

string AISerializeExpression(const Expression &expr) {
	MemoryStream stream;
	BinarySerializer::Serialize(expr, stream);
	const string bytes(const_char_ptr_cast(stream.GetData()), stream.GetPosition());
	return Blob::ToBase64(string_t(bytes));
}

unique_ptr<Expression> AIDeserializeExpression(ClientContext &context, const string &base64) {
	string bytes = Blob::FromBase64(string_t(base64));
	MemoryStream stream(data_ptr_cast(&bytes[0]), bytes.size());
	BinaryDeserializer deserializer(stream);
	deserializer.Set<ClientContext &>(context);
	deserializer.Begin();
	auto expr = Expression::Deserialize(deserializer);
	deserializer.End();
	return expr;
}

bool AIDetectMixedLeaf(const Expression &expr, AIMixedLeaf &out) {
	if (AIIsFilterLeaf(expr)) {
		out = AIMixedLeaf {'F', 0, &expr, nullptr, {}};
		return true;
	}
	// Single comparison: <ai_call> <op> <const>.
	if (BoundComparisonExpression::IsComparison(expr)) {
		auto &cmp = expr.Cast<BoundFunctionExpression>();
		const Expression &lhs = BoundComparisonExpression::Left(cmp);
		const Expression &rhs = BoundComparisonExpression::Right(cmp);
		const char lk = AIComparableKind(lhs);
		const char rk = AIComparableKind(rhs);
		const Expression *ai = nullptr;
		const Expression *cst = nullptr;
		bool flipped = false;
		char kind = 0;
		if (lk && rhs.IsFoldable()) {
			ai = &lhs;
			cst = &rhs;
			kind = lk;
		} else if (rk && lhs.IsFoldable()) {
			ai = &rhs;
			cst = &lhs;
			kind = rk;
			flipped = true;
		}
		if (kind) {
			const char op = AICmpOp(expr.GetExpressionType(), flipped);
			if (op) {
				out = AIMixedLeaf {kind, op, ai, cst, {}};
				return true;
			}
		}
		return AIDetectWrappedLeaf(expr, out); // a comparison around the call, e.g. lower(ai_complete(x)) = 'yes'
	}
	// Set membership: `ai_classify(x) IN (...)` desugars to OR of same-call equalities.
	{
		const Expression *ai = nullptr;
		char kind = 0;
		vector<const Expression *> csts;
		if (AIDetectSameCallEqualities(expr, ai, kind, csts)) {
			out = AIMixedLeaf {kind, 'i', ai, nullptr, std::move(csts)};
			return true;
		}
	}
	// NOT(single =) -> != ; NOT(OR of same-call equalities) -> NOT IN. (`x NOT IN (...)` desugars this way.)
	if (expr.GetExpressionType() == ExpressionType::OPERATOR_NOT) {
		auto &op = expr.Cast<BoundOperatorExpression>();
		if (op.GetChildren().size() == 1) {
			const Expression &child = *op.GetChildren()[0];
			AIMixedLeaf inner;
			if (AIDetectMixedLeaf(child, inner) && inner.op == 'e') {
				out = inner;
				out.op = 'n'; // NOT(x = c)  ==  x != c  (classify/complete are non-null)
				return true;
			}
			const Expression *ai = nullptr;
			char kind = 0;
			vector<const Expression *> csts;
			if (AIDetectSameCallEqualities(child, ai, kind, csts)) {
				out = AIMixedLeaf {kind, 'I', ai, nullptr, std::move(csts)};
				return true;
			}
		}
	}
	return AIDetectWrappedLeaf(expr, out);
}

bool AIIsMixedBoolean(const Expression &expr) {
	AIMixedLeaf leaf;
	if (AIDetectMixedLeaf(expr, leaf)) {
		return true;
	}
	const auto type = expr.GetExpressionType();
	if (type == ExpressionType::CONJUNCTION_AND || type == ExpressionType::CONJUNCTION_OR) {
		auto &conj = expr.Cast<BoundConjunctionExpression>();
		if (conj.GetChildren().empty()) {
			return false;
		}
		for (auto &child : conj.GetChildren()) {
			if (!AIIsMixedBoolean(*child)) {
				return false;
			}
		}
		return true;
	}
	if (type == ExpressionType::OPERATOR_NOT) {
		auto &op = expr.Cast<BoundOperatorExpression>();
		return op.GetChildren().size() == 1 && AIIsMixedBoolean(*op.GetChildren()[0]);
	}
	return false;
}

idx_t AICountMixedLeaves(const Expression &expr) {
	AIMixedLeaf leaf;
	if (AIDetectMixedLeaf(expr, leaf)) {
		return 1;
	}
	idx_t n = 0;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) { n += AICountMixedLeaves(child); });
	return n;
}

bool AIAnyNonFilterLeaf(const Expression &expr) {
	AIMixedLeaf leaf;
	if (AIDetectMixedLeaf(expr, leaf)) {
		return leaf.kind != 'F' || leaf.op == 'w'; // a wrapped ai_filter is not a plain leaf either
	}
	bool any = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (AIAnyNonFilterLeaf(child)) {
			any = true;
		}
	});
	return any;
}

//! Two mixed leaves are the same predicate when they ask the same AI call the same question --
//! same kind, same comparison, same constant(s). Compared by expression, not by position.
static bool SameMixedLeaf(const AIMixedLeaf &a, const AIMixedLeaf &b) {
	if (a.kind != b.kind || a.op != b.op || a.cst_set.size() != b.cst_set.size()) {
		return false;
	}
	if (!a.ai_call || !b.ai_call || !a.ai_call->Equals(*b.ai_call)) {
		return false;
	}
	if ((a.cst == nullptr) != (b.cst == nullptr) || (a.wrapper == nullptr) != (b.wrapper == nullptr)) {
		return false;
	}
	if (a.cst && !a.cst->Equals(*b.cst)) {
		return false;
	}
	if (a.wrapper && !a.wrapper->Equals(*b.wrapper)) {
		return false;
	}
	for (idx_t i = 0; i < a.cst_set.size(); i++) {
		if (!a.cst_set[i] || !b.cst_set[i] || !a.cst_set[i]->Equals(*b.cst_set[i])) {
			return false;
		}
	}
	return true;
}

static unique_ptr<AIFilterTreeNode> BuildMixedTreeRec(const Expression &expr, vector<AIMixedLeaf> &leaves) {
	AIMixedLeaf leaf;
	if (AIDetectMixedLeaf(expr, leaf)) {
		for (idx_t i = 0; i < leaves.size(); i++) {
			if (SameMixedLeaf(leaves[i], leaf)) {
				return AIFilterTreeNode::Leaf(i);
			}
		}
		const idx_t idx = leaves.size();
		leaves.push_back(leaf);
		return AIFilterTreeNode::Leaf(idx);
	}
	const auto type = expr.GetExpressionType();
	if (type == ExpressionType::CONJUNCTION_AND || type == ExpressionType::CONJUNCTION_OR) {
		auto &conj = expr.Cast<BoundConjunctionExpression>();
		const auto tree_type =
		    type == ExpressionType::CONJUNCTION_AND ? AIFilterTreeType::AND_OP : AIFilterTreeType::OR_OP;
		vector<unique_ptr<AIFilterTreeNode>> children;
		for (auto &child : conj.GetChildren()) {
			auto sub = BuildMixedTreeRec(*child, leaves);
			if (sub->type == tree_type) {
				for (auto &grandchild : sub->children) {
					children.push_back(std::move(grandchild));
				}
			} else {
				children.push_back(std::move(sub));
			}
		}
		return AIFilterTreeNode::Op(tree_type, std::move(children));
	}
	auto &op = expr.Cast<BoundOperatorExpression>();
	vector<unique_ptr<AIFilterTreeNode>> children;
	children.push_back(BuildMixedTreeRec(*op.GetChildren()[0], leaves));
	return AIFilterTreeNode::Op(AIFilterTreeType::NOT_OP, std::move(children));
}

unique_ptr<AIFilterTreeNode> AIBuildMixedTree(const Expression &expr, vector<AIMixedLeaf> &leaves) {
	return NormalizeSubtree(BuildMixedTreeRec(expr, leaves));
}

bool AIBuildMixedLeafArgs(ClientContext &context, const vector<AIMixedLeaf> &leaves,
                          vector<unique_ptr<Expression>> &call_prompts, vector<unique_ptr<Expression>> &feat_pred,
                          vector<unique_ptr<Expression>> &feat_input, string &meta_str) {
	const idx_t m = leaves.size();
	call_prompts.resize(m);
	feat_pred.resize(m);
	feat_input.resize(m);
	vector<string> metas(m);
	for (idx_t j = 0; j < m; j++) {
		auto &leaf = leaves[j];
		auto &children = leaf.ai_call->Cast<BoundFunctionExpression>().GetChildren();
		// Wrapped leaf: the value slot carries the serialized wrapper (base64: no ';', ':' or '@', so the
		// token still splits like the others); the feature text names the wrapper over `result`.
		const bool wrapped = leaf.op == 'w';
		const string wrapper_text = wrapped ? leaf.wrapper->ToString() : string();
		if (leaf.kind == 'F') {
			// ai_filter(p): call prompt is p; feature = AISplitPrompt(p) -- the plain question even when
			// wrapped, since that text is also the typed backend's question; meta = "F".
			call_prompts[j] = children[0]->Copy();
			auto split = AISplitPrompt(context, *children[0]);
			feat_pred[j] = std::move(split.first);
			feat_input[j] = std::move(split.second);
			metas[j] = wrapped ? "F:wrap:" + AISerializeExpression(*leaf.wrapper) : "F";
			continue;
		}
		// key_start = count of DECLARED args (excludes AIKeyBind's hidden history keys). v1 handles 2-arg
		// ai_classify(input, cats), 1/2-arg ai_score(input[, criteria]), 1-arg ai_complete(input); anything
		// else bails (safe: the caller leaves the plan untouched).
		const idx_t key_start = children.size() - AICountColRefs(*children[0]);
		if ((leaf.kind == 'C' && key_start != 2) ||
		    (leaf.kind == 'S' && key_start != 1 && key_start != 2 && key_start != 4) ||
		    (leaf.kind == 'M' && key_start != 1)) {
			return false;
		}
		// IN ('i') / NOT IN ('I'): the value is the ','-joined set of members; else the single constant.
		const bool is_set = (leaf.op == 'i' || leaf.op == 'I');
		string val;
		if (wrapped) {
			val = AISerializeExpression(*leaf.wrapper);
		} else if (is_set) {
			for (idx_t k = 0; k < leaf.cst_set.size(); k++) {
				Value v;
				try {
					v = ExpressionExecutor::EvaluateScalar(context, *leaf.cst_set[k]);
				} catch (...) {
					return false;
				}
				if (k) {
					val += ",";
				}
				val += v.IsNull() ? string() : StringValue::Get(v);
			}
		} else {
			Value cst_val;
			try {
				cst_val = ExpressionExecutor::EvaluateScalar(context, *leaf.cst);
			} catch (...) {
				return false;
			}
			val = leaf.kind == 'S' ? cst_val.ToString() : (cst_val.IsNull() ? string() : StringValue::Get(cst_val));
		}
		metas[j] = string(1, leaf.kind) + ":" + (wrapped ? "wrap" : AIOpName(leaf.op)) + ":" + val;

		auto input = children[0]->Copy();
		if (leaf.kind == 'C') {
			Value cats_val;
			try {
				cats_val = ExpressionExecutor::EvaluateScalar(context, *children[1]);
			} catch (...) {
				return false;
			}
			string cats;
			for (auto &c : ListValue::GetChildren(cats_val)) {
				if (!cats.empty()) {
					cats += ", ";
				}
				cats += StringValue::Get(c);
			}
			vector<unique_ptr<Expression>> parts;
			parts.push_back(make_uniq<BoundConstantExpression>(Value("Categories: " + cats + "\n\nInput:\n")));
			parts.push_back(input->Copy());
			parts.push_back(make_uniq<BoundConstantExpression>(
			    Value(string("\n\nRespond with exactly one of the listed categories, verbatim."))));
			call_prompts[j] = BuildConcat(context, std::move(parts));
			feat_pred[j] = make_uniq<BoundConstantExpression>(Value(
			    wrapped ? ("Does the classification satisfy " + wrapper_text + "? Categories: " + cats)
			    : is_set
			        ? ("Is the classification " + string(AIOpPhrase(leaf.op)) + " " + val + "? Categories: " + cats)
			        : ("Is the classification '" + val + "'? Categories: " + cats)));
			feat_input[j] = std::move(input);
		} else if (leaf.kind == 'S') {
			string crit;
			if (key_start >= 2) {
				Value crit_val;
				try {
					crit_val = ExpressionExecutor::EvaluateScalar(context, *children[1]);
				} catch (...) {
					return false;
				}
				crit = crit_val.IsNull() ? string() : StringValue::Get(crit_val);
			}
			// Ranged ai_score(input, criteria, lo, hi): bake the range into the prompt + carry lo,hi,type in
			// the meta (@lo,hi,i|d) so the node clamps/rounds identically before comparing. Matches the scalar.
			bool has_range = false;
			double lo = 0, hi = 1;
			bool is_int = false;
			if (key_start >= 4) {
				is_int = children[2]->GetReturnType().IsIntegral();
				try {
					lo = ExpressionExecutor::EvaluateScalar(context, *children[2]).GetValue<double>();
					hi = ExpressionExecutor::EvaluateScalar(context, *children[3]).GetValue<double>();
					has_range = true;
				} catch (...) {
					return false;
				}
			}
			string prefix;
			if (has_range) {
				prefix = "Provide a score from " + AIFormatBound(lo, is_int) + " to " + AIFormatBound(hi, is_int) +
				         (is_int ? " as an integer." : ".") + "\n\nScoring criteria: " + crit + "\n\nInput:\n";
			} else if (!crit.empty()) {
				prefix = "Scoring criteria: " + crit + "\n\nInput:\n";
			}
			if (prefix.empty()) {
				call_prompts[j] = input->Copy();
			} else {
				vector<unique_ptr<Expression>> parts;
				parts.push_back(make_uniq<BoundConstantExpression>(Value(prefix)));
				parts.push_back(input->Copy());
				call_prompts[j] = BuildConcat(context, std::move(parts));
			}
			if (has_range) {
				metas[j] +=
				    "@" + AIFormatBound(lo, is_int) + "," + AIFormatBound(hi, is_int) + "," + (is_int ? "i" : "d");
			}
			feat_pred[j] = make_uniq<BoundConstantExpression>(
			    Value((wrapped ? "Does the score satisfy " + wrapper_text
			                   : string("Is the score ") + AIOpPhrase(leaf.op) + " " + val) +
			          (crit.empty() ? string() : " for: " + crit) + "?"));
			feat_input[j] = std::move(input);
		} else { // 'M' ai_complete: raw completion compared to a literal (or set)
			call_prompts[j] = input->Copy();
			feat_pred[j] = make_uniq<BoundConstantExpression>(
			    Value(wrapped  ? ("Does the completion satisfy " + wrapper_text + "?")
			          : is_set ? ("Is the completion " + string(AIOpPhrase(leaf.op)) + " " + val + "?")
			                   : ("Is the completion '" + val + "'?")));
			feat_input[j] = std::move(input);
		}
		if (!call_prompts[j]) {
			return false;
		}
	}
	meta_str.clear();
	for (idx_t j = 0; j < m; j++) {
		if (j) {
			meta_str += ";";
		}
		meta_str += metas[j];
	}
	return true;
}

unique_ptr<Expression> AIBuildMixedNode(ClientContext &context, const AIFilterTreeNode &tree,
                                        const vector<AIMixedLeaf> &leaves, bool speculative) {
	vector<unique_ptr<Expression>> call_prompts, feat_pred, feat_input;
	string meta_str;
	if (!AIBuildMixedLeafArgs(context, leaves, call_prompts, feat_pred, feat_input, meta_str)) {
		return nullptr;
	}
	const idx_t m = leaves.size();
	vector<unique_ptr<Expression>> args;
	args.reserve(2 + 3 * m);
	args.push_back(make_uniq<BoundConstantExpression>(Value(AIFilterTreeSerialize(tree))));
	for (idx_t j = 0; j < m; j++) {
		args.push_back(std::move(call_prompts[j]));
	}
	for (idx_t j = 0; j < m; j++) {
		args.push_back(std::move(feat_pred[j]));
	}
	for (idx_t j = 0; j < m; j++) {
		args.push_back(std::move(feat_input[j]));
	}
	args.push_back(make_uniq<BoundConstantExpression>(Value(meta_str)));
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto &entry = catalog.GetEntry<ScalarFunctionCatalogEntry>(
	    context, QualifiedName(catalog.GetName(), Identifier::DefaultSchema(),
	                           speculative ? "speculative_ai_function_with_embed" : "ai_function_with_embed"));
	FunctionBinder function_binder(context);
	ErrorData error;
	return function_binder.BindScalarFunction(entry, std::move(args), error);
}

} // namespace duckdb
