//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/semantic_filter_pullup.hpp
//
// Pull ai_filter predicates up above the inner-join cluster (so the expensive LLM filter runs on
// the joined + Yannakakis-reduced result), leaving one combined speculative_ai_function_with_embed
// MLP-gated pre-filter at the original pushed-down position (per leaf filter) that prunes
// likely-to-fail rows before the join. Opt-in via the DUCKDB_SEMANTIC_PULLUP environment variable.
//===----------------------------------------------------------------------===//

#pragma once

#include "optimizer/ai_filter_tree_build.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class SemanticFilterPullup {
public:
	explicit SemanticFilterPullup(Optimizer &optimizer) : optimizer(optimizer) {
	}
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	Optimizer &optimizer;

	//! Is `op` an inner-join cluster node (inner comparison join or cross product)?
	static bool IsClusterNode(const LogicalOperator &op);
	//! Find each outermost inner-join cluster and pull its ai_filters up above it.
	void PullupTree(unique_ptr<LogicalOperator> &op_slot);
	//! Within a cluster subtree, move each AI-comparison predicate (ai_filter / ai_classify=/ai_score>/
	//! ai_complete=) out into `pulled`, leaving one combined speculative_ai_function_with_embed pre-filter
	//! per leaf FILTER they came from.
	void ExtractAIFilters(unique_ptr<LogicalOperator> &op, vector<unique_ptr<Expression>> &pulled,
	                      idx_t cluster_card);
	//! Build one bound speculative_ai_function_with_embed over the AND of the pulled leaves (baking each
	//! leaf's scalar-exact call prompt + meta; the node embeds predicate/input internally for the MLP
	//! feature). Reads the leaves' expression pointers, so call it BEFORE the leaf exprs are moved away.
	unique_ptr<Expression> BuildSpeculativeCall(vector<AIMixedLeaf> &leaves);
};

} // namespace duckdb
