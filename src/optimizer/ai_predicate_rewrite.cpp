#include "optimizer/ai_predicate_rewrite.hpp"

#include "ai_settings.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/common/error_data.hpp"
#include "filter_tree_order.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/function_binder.hpp"
#include "optimizer/ai_filter_tree_build.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"

#include <cstdlib>

namespace duckdb {

unique_ptr<Expression> AIPredicateRewrite::BuildPredicateCall(vector<unique_ptr<Expression>> args) {
	auto &context = optimizer.context;
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto &entry = catalog.GetEntry<ScalarFunctionCatalogEntry>(
	    context, QualifiedName(catalog.GetName(), Identifier::DefaultSchema(), "ai_function_with_embed"));
	FunctionBinder function_binder(context);
	ErrorData error;
	return function_binder.BindScalarFunction(entry, std::move(args), error);
}

bool AIPredicateRewrite::RewriteMixedFilter(LogicalFilter &filter) {
	auto &context = optimizer.context;

	// Fire only for a boolean tree over >=2 AI leaves with >=1 non-ai_filter leaf. All-ai_filter falls
	// through to RewriteFilter's original (tested, unchanged) pure path.
	idx_t total = 0;
	bool any_non_filter = false;
	bool any_convertible = false;
	for (auto &expr : filter.expressions) {
		if (AIIsMixedBoolean(*expr)) {
			any_convertible = true;
			total += AICountMixedLeaves(*expr);
			if (AIAnyNonFilterLeaf(*expr)) {
				any_non_filter = true;
			}
		}
	}
	if (!any_convertible || total < 2 || !any_non_filter) {
		return false;
	}

	// One combined tree over the convertible predicates (ANDed) + leaves in tree order.
	vector<AIMixedLeaf> leaves;
	vector<unique_ptr<AIFilterTreeNode>> subtrees;
	for (auto &expr : filter.expressions) {
		if (AIIsMixedBoolean(*expr)) {
			subtrees.push_back(AIBuildMixedTree(*expr, leaves));
		}
	}
	unique_ptr<AIFilterTreeNode> tree =
	    subtrees.size() == 1 ? std::move(subtrees[0])
	                         : AIFilterTreeNode::Op(AIFilterTreeType::AND_OP, std::move(subtrees));
	const string tree_str = AIFilterTreeSerialize(*tree);
	const idx_t m = leaves.size();

	// Per leaf: the exact scalar-matching call prompt, the (feat_pred, feat_input) split, and the meta
	// token -- shared with the speculative pull-up via ai_filter_tree_build.
	vector<unique_ptr<Expression>> call_prompts, feat_pred, feat_input;
	string meta_str;
	if (!AIBuildMixedLeafArgs(context, leaves, call_prompts, feat_pred, feat_input, meta_str)) {
		return false; // a leaf was not bakeable -> leave the plan untouched
	}

	// Non-convertible predicates stay as separate filter expressions.
	vector<unique_ptr<Expression>> keep_exprs;
	for (auto &expr : filter.expressions) {
		if (!AIIsMixedBoolean(*expr)) {
			keep_exprs.push_back(std::move(expr));
		}
	}

	// ai_function_with_embed(tree_str, call_0..m-1, pred_0..m-1, input_0..m-1, meta_str).
	vector<unique_ptr<Expression>> args;
	args.reserve(2 + 3 * m);
	args.push_back(make_uniq<BoundConstantExpression>(Value(tree_str)));
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
	auto node = BuildPredicateCall(std::move(args));
	if (!node) {
		return false; // bind failed: leave the plan untouched
	}

	filter.expressions.clear();
	filter.expressions.push_back(std::move(node));
	for (auto &expr : keep_exprs) {
		filter.expressions.push_back(std::move(expr));
	}
	filter.ResolveOperatorTypes();
	return true;
}

void AIPredicateRewrite::RewriteFilter(unique_ptr<LogicalOperator> &filter_slot) {
	auto &filter = filter_slot->Cast<LogicalFilter>();

	// Mixed AI-comparison tree (>=2 leaves, >=1 non-ai_filter) -> generalized node; else fall through to
	// the pure-ai_filter path below.
	if (RewriteMixedFilter(filter)) {
		return;
	}

	// A predicate is convertible when it is a pure ai_filter boolean tree. Only rewrite when there are
	// >=2 ai_filter leaves across the convertible predicates -- a lone ai_filter gains nothing.
	idx_t ai_leaf_count = 0;
	bool any_convertible = false;
	for (auto &expr : filter.expressions) {
		if (AIIsPureFilterBoolean(*expr)) {
			any_convertible = true;
			ai_leaf_count += AICountFilterLeaves(*expr);
		}
	}
	if (!any_convertible || ai_leaf_count < 2) {
		return;
	}

	// Build one combined tree (convertible predicates ANDed together) and copy the leaf prompts.
	vector<unique_ptr<Expression>> leaf_prompts;
	vector<unique_ptr<AIFilterTreeNode>> subtrees;
	for (auto &expr : filter.expressions) {
		if (AIIsPureFilterBoolean(*expr)) {
			subtrees.push_back(AIBuildFilterTree(*expr, leaf_prompts));
		}
	}
	unique_ptr<AIFilterTreeNode> tree = subtrees.size() == 1
	                                        ? std::move(subtrees[0])
	                                        : AIFilterTreeNode::Op(AIFilterTreeType::AND_OP, std::move(subtrees));
	const string tree_str = AIFilterTreeSerialize(*tree);
	const idx_t m = leaf_prompts.size();

	// Non-convertible predicates stay as separate filter expressions.
	vector<unique_ptr<Expression>> keep_exprs;
	for (auto &expr : filter.expressions) {
		if (!AIIsPureFilterBoolean(*expr)) {
			keep_exprs.push_back(std::move(expr));
		}
	}

	// Split each leaf's prompt into (predicate, input) text. The node embeds them internally (lazily,
	// overlapped with the LLM calls), so no projection or column remapping is needed -- the text
	// expressions reference the child's columns directly, right inside the filter.
	vector<unique_ptr<Expression>> pred_prompts(m), input_prompts(m);
	for (idx_t j = 0; j < m; j++) {
		auto split = AISplitPrompt(optimizer.context, *leaf_prompts[j]);
		pred_prompts[j] = std::move(split.first);
		input_prompts[j] = std::move(split.second);
	}

	// ai_function_with_embed(tree_str, prompt_0..{m-1}, pred_text_0..{m-1}, input_text_0..{m-1}) -- all
	// VARCHAR. The full prompt is what the LLM ai_filter sees; predicate/input are embedded inside the
	// node to build the MLP feature [pred_emb, input_emb, cos_sim].
	vector<unique_ptr<Expression>> pred_args;
	pred_args.reserve(1 + 3 * m);
	pred_args.push_back(make_uniq<BoundConstantExpression>(Value(tree_str)));
	for (idx_t j = 0; j < m; j++) {
		pred_args.push_back(std::move(leaf_prompts[j]));
	}
	for (idx_t j = 0; j < m; j++) {
		pred_args.push_back(std::move(pred_prompts[j]));
	}
	for (idx_t j = 0; j < m; j++) {
		pred_args.push_back(std::move(input_prompts[j]));
	}
	auto pred_expr = BuildPredicateCall(std::move(pred_args));
	if (!pred_expr) {
		return; // bind failed: leave the plan untouched
	}

	// The filter's predicate becomes ai_function_with_embed plus any non-convertible expressions.
	filter.expressions.clear();
	filter.expressions.push_back(std::move(pred_expr));
	for (auto &expr : keep_exprs) {
		filter.expressions.push_back(std::move(expr));
	}
	filter_slot->ResolveOperatorTypes();
}

unique_ptr<Expression> AIPredicateRewrite::TryBuildReorderNode(const Expression &tree_expr) {
	// A single boolean expression that is an AI tree over >=2 leaves -> one ai_function_with_embed node.
	if (!AIIsMixedBoolean(tree_expr) || AICountMixedLeaves(tree_expr) < 2) {
		return nullptr;
	}
	vector<AIMixedLeaf> leaves;
	auto tree = AIBuildMixedTree(tree_expr, leaves);
	const string tree_str = AIFilterTreeSerialize(*tree);
	const idx_t m = leaves.size();

	vector<unique_ptr<Expression>> call_prompts, feat_pred, feat_input;
	string meta_str;
	if (!AIBuildMixedLeafArgs(optimizer.context, leaves, call_prompts, feat_pred, feat_input, meta_str)) {
		return nullptr; // a leaf was not bakeable
	}
	vector<unique_ptr<Expression>> args;
	args.reserve(2 + 3 * m);
	args.push_back(make_uniq<BoundConstantExpression>(Value(tree_str)));
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
	return BuildPredicateCall(std::move(args)); // null on bind failure -> caller leaves the tree untouched
}

void AIPredicateRewrite::RewriteCaseInExpression(unique_ptr<Expression> &expr) {
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_CASE) {
		auto &case_expr = expr->Cast<BoundCaseExpression>();
		for (auto &check : case_expr.CaseChecksMutable()) {
			// The WHEN condition is boolean; folding its AI tree into a node keeps it boolean, so the
			// node's per-row/cost-aware ordering + short-circuit replace DuckDB's adaptive-filter Select.
			if (auto node = TryBuildReorderNode(*check.when_expr)) {
				check.when_expr = std::move(node);
			} else {
				RewriteCaseInExpression(check.when_expr); // e.g. a nested CASE inside the condition
			}
			RewriteCaseInExpression(check.then_expr);
		}
		RewriteCaseInExpression(case_expr.ElseMutable());
		return;
	}
	ExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<Expression> &child) { RewriteCaseInExpression(child); });
}

void AIPredicateRewrite::RewriteCaseInOperators(unique_ptr<LogicalOperator> &op) {
	for (auto &child : op->children) {
		RewriteCaseInOperators(child);
	}
	for (auto &expr : op->expressions) {
		RewriteCaseInExpression(expr);
	}
}

void AIPredicateRewrite::CollectFilters(unique_ptr<LogicalOperator> &op,
                                        vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> &out) {
	if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
		// Count AI leaves (ai_filter + ai_classify/ai_score/ai_complete comparisons); >=2 -> a reorder candidate.
		idx_t ai_leaf_count = 0;
		for (auto &expr : op->expressions) {
			if (AIIsMixedBoolean(*expr)) {
				ai_leaf_count += AICountMixedLeaves(*expr);
			}
		}
		if (ai_leaf_count >= 2) {
			out.push_back(op);
		}
	}
	for (auto &child : op->children) {
		CollectFilters(child, out);
	}
}

unique_ptr<LogicalOperator> AIPredicateRewrite::Optimize(unique_ptr<LogicalOperator> op) {
	if (!AIBoolSetting(optimizer.context, "ai_reorder", true)) {
		return op;
	}
	vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> filters;
	CollectFilters(op, filters);
	for (auto &filter_slot : filters) {
		RewriteFilter(filter_slot.get());
	}
	// Also fold AI boolean trees inside CASE WHEN conditions (anywhere: projection / aggregate / a CASE
	// used as a filter predicate). Runs after the filter rewrite so a filter already turned into a node is
	// simply skipped (it is not a CASE).
	RewriteCaseInOperators(op);
	return op;
}

} // namespace duckdb
