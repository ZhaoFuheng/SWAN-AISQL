//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/ai_embed_filter.hpp
//
// Rewrite rule: below every ai_filter, insert a projection that computes
// ai_embed(prompt), so the prompt embedding is materialized ahead of the filter.
// The embedding is the input feature for (future) filter-selectivity prediction.
// The extra column is projected away by the filter, so query results are unchanged.
// Opt-in via the DUCKDB_AI_EMBED_FILTER environment variable.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"

#include <functional>

namespace duckdb {

class AIEmbedFilter {
public:
	explicit AIEmbedFilter(Optimizer &optimizer) : optimizer(optimizer) {
	}
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	Optimizer &optimizer;

	//! Collect every LogicalFilter (as a mutable slot) whose predicate calls ai_filter.
	void CollectAIFilters(unique_ptr<LogicalOperator> &op,
	                      vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> &out);
	//! Insert an ai_embed(prompt) projection below `filter`, remapping references above it.
	void RewriteFilter(LogicalOperator &root, unique_ptr<LogicalOperator> &filter_slot);
	//! Build a bound ai_embed(prompt) scalar call (prompt is copied).
	unique_ptr<Expression> BuildEmbedCall(const Expression &prompt);
};

} // namespace duckdb
