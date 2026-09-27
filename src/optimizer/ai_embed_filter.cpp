#include "optimizer/ai_embed_filter.hpp"

#include "ai_settings.hpp"
#include "duckdb/planner/binder.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/common/projection_index.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

#include <cstdlib>

namespace duckdb {

//! Collect the prompt argument (child 0) of every call to `fname` inside `expr`.
static void CollectPromptsByName(const Expression &expr, const char *fname, vector<const Expression *> &prompts) {
	if (expr.GetExpressionType() == ExpressionType::BOUND_FUNCTION) {
		auto &fn = expr.Cast<BoundFunctionExpression>();
		if (fn.Function().GetName() == fname && !fn.GetChildren().empty()) {
			prompts.push_back(fn.GetChildren()[0].get());
		}
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](const Expression &child) { CollectPromptsByName(child, fname, prompts); });
}

//! Prompts in `filter` worth embedding. The embedding only pays off when a selectivity consumer
//! exists: several ai_filters in one filter (to order them) or a speculative_ai_filter (to feed its
//! pre-reduction estimate). A lone ai_filter (e.g. `WHERE ai_filter(...)`) gets nothing -- there is
//! no decision to inform, so the per-row embedding would be pure overhead.
static void CollectEmbedPrompts(const LogicalFilter &filter, vector<const Expression *> &prompts) {
	vector<const Expression *> ai_prompts;
	vector<const Expression *> speculative_prompts;
	for (auto &expr : filter.expressions) {
		CollectPromptsByName(*expr, "ai_filter", ai_prompts);
		CollectPromptsByName(*expr, "speculative_ai_filter", speculative_prompts);
	}
	if (ai_prompts.size() >= 2) {
		for (auto *p : ai_prompts) {
			prompts.push_back(p);
		}
	}
	for (auto *p : speculative_prompts) {
		prompts.push_back(p);
	}
}

unique_ptr<Expression> AIEmbedFilter::BuildEmbedCall(const Expression &prompt) {
	auto &context = optimizer.context;
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto &entry = catalog.GetEntry<ScalarFunctionCatalogEntry>(
	    context, QualifiedName(catalog.GetName(), Identifier::DefaultSchema(), "ai_embed"));
	auto embed_fn = entry.functions.GetFunctionByArguments(context, {LogicalType::VARCHAR});
	vector<unique_ptr<Expression>> args;
	args.push_back(prompt.Copy());
	FunctionBinder function_binder(context);
	return function_binder.BindScalarFunction(embed_fn, std::move(args));
}

void AIEmbedFilter::RewriteFilter(LogicalOperator &root, unique_ptr<LogicalOperator> &filter_slot) {
	auto &filter = filter_slot->Cast<LogicalFilter>();
	vector<const Expression *> prompts;
	CollectEmbedPrompts(filter, prompts);
	if (prompts.empty()) {
		return;
	}

	auto &child = filter.children[0];
	child->ResolveOperatorTypes();
	auto child_bindings = child->GetColumnBindings();
	auto &child_types = child->types;
	if (child_bindings.size() != child_types.size()) {
		return; // defensive: mismatched schema, skip
	}

	// Projection select list: pass every child column through, then one ai_embed(prompt) per ai_filter.
	vector<unique_ptr<Expression>> select_list;
	select_list.reserve(child_bindings.size() + prompts.size());
	for (idx_t i = 0; i < child_bindings.size(); i++) {
		select_list.push_back(make_uniq<BoundColumnRefExpression>(child_types[i], child_bindings[i]));
	}
	for (auto *prompt : prompts) {
		select_list.push_back(BuildEmbedCall(*prompt));
	}

	const auto proj_index = optimizer.binder.GenerateTableIndex();
	auto projection = make_uniq<LogicalProjection>(proj_index, std::move(select_list));
	projection->children.push_back(std::move(child));
	projection->ResolveOperatorTypes();

	// Splice the projection in as the filter's new child. Drop the embed columns from the filter's
	// output via a projection map, so nothing above the filter sees the extra vectors. An existing
	// projection map already indexes the passthrough positions (0..k-1) and excludes the embed
	// columns appended at the end, so it stays correct unchanged; only add one when absent.
	filter.children[0] = std::move(projection);
	if (!filter.HasProjectionMap()) {
		for (idx_t i = 0; i < child_bindings.size(); i++) {
			filter.projection_map.emplace_back(ProjectionIndex(i));
		}
	}
	filter_slot->ResolveOperatorTypes();

	// Everything at/above the filter that referenced the old child columns must now reference the
	// projection's pass-through columns. The projection's own subtree keeps the old bindings.
	ColumnBindingReplacer replacer;
	for (idx_t i = 0; i < child_bindings.size(); i++) {
		replacer.replacement_bindings.emplace_back(child_bindings[i], ColumnBinding(proj_index, ProjectionIndex(i)));
	}
	replacer.stop_operator = filter.children[0].get();
	replacer.VisitOperator(root);
}

void AIEmbedFilter::CollectAIFilters(unique_ptr<LogicalOperator> &op,
                                     vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> &out) {
	if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
		vector<const Expression *> prompts;
		CollectEmbedPrompts(op->Cast<LogicalFilter>(), prompts);
		if (!prompts.empty()) {
			out.push_back(op);
		}
	}
	for (auto &child : op->children) {
		CollectAIFilters(child, out);
	}
}

unique_ptr<LogicalOperator> AIEmbedFilter::Optimize(unique_ptr<LogicalOperator> op) {
	if (!AIBoolSetting(optimizer.context, "ai_debug_embed_filter", false)) {
		return op; // opt-in only; leave every other query untouched
	}
	vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> filters;
	CollectAIFilters(op, filters);
	for (auto &filter_slot : filters) {
		RewriteFilter(*op, filter_slot.get());
	}
	return op;
}

} // namespace duckdb
