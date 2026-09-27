//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/ai_predicate_rewrite.hpp
//
// Rewrite a WHERE predicate that is a boolean tree (AND/OR/NOT) over >=2 ai_filter leaves into a
// single ai_predicate(tree, prompts..., embeddings...) call, materializing one ai_embed per leaf
// below the filter. ai_predicate chooses the evaluation order PER ROW from the embeddings so the
// expensive LLM calls short-circuit early. Opt-in via the DUCKDB_AI_REORDER environment variable.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"

#include <functional>
#include <utility>

namespace duckdb {

class LogicalFilter;

class AIPredicateRewrite {
public:
	explicit AIPredicateRewrite(Optimizer &optimizer) : optimizer(optimizer) {
	}
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	Optimizer &optimizer;

	void CollectFilters(unique_ptr<LogicalOperator> &op,
	                    vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> &out);
	void RewriteFilter(unique_ptr<LogicalOperator> &filter_slot);
	//! Generalized path: a boolean tree over >=2 AI leaves with >=1 non-ai_filter leaf (ai_classify=/
	//! ai_score>/ai_complete= comparisons) -> one ai_function_with_embed node carrying per-leaf meta.
	//! Returns true if it rewrote the filter; false (all-ai_filter or not applicable) falls through to
	//! the pure-ai_filter path in RewriteFilter.
	bool RewriteMixedFilter(unique_ptr<LogicalOperator> &filter_slot);
	//! Install the folded node: alone in the filter when nothing else is left, otherwise in a NEW filter
	//! ABOVE the relational remainder, so the cheap predicates always run first (see the .cpp note).
	void InstallAINode(unique_ptr<LogicalOperator> &filter_slot, unique_ptr<Expression> node,
	                   vector<unique_ptr<Expression>> keep_exprs);
	//! Build one ai_function_with_embed node from a SINGLE boolean expression that is an AI tree over >=2
	//! leaves (ai_filter / ai_classify=,IN / ai_score-compare / ai_complete=). Returns null if the
	//! expression is not such a tree or a leaf is not bakeable. Used for CASE WHEN conditions.
	unique_ptr<Expression> TryBuildReorderNode(const Expression &tree_expr);
	//! Walk every operator's expressions and fold an AI boolean tree found in a CASE WHEN condition into a
	//! reorder node (the WHEN stays boolean). Lets the per-row/cost-aware node + short-circuit apply inside
	//! CASE, not just in a WHERE filter.
	void RewriteCaseInOperators(unique_ptr<LogicalOperator> &op);
	void RewriteCaseInExpression(unique_ptr<Expression> &expr);
	unique_ptr<Expression> BuildPredicateCall(vector<unique_ptr<Expression>> args);
};

} // namespace duckdb
