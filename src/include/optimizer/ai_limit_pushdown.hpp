//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/ai_limit_pushdown.hpp
//
// Push a plain LIMIT k that sits directly above an AI boolean-tree filter INTO the ai_function_with_embed
// node (building it from the raw ai_filter / ai_classify=,IN / ai_score-compare / ai_complete= leaves if
// needed), so it stops making LLM calls once k rows pass instead of evaluating the whole scan. Sound
// because the LIMIT keeps only k rows, so the still-unevaluated rows can stay false -- but ONLY when the
// node decides the whole filter: a predicate left above the node would filter below k, so the push is
// skipped unless the node is the sole filter predicate. NOT applied to ORDER BY ... LIMIT (top-K needs
// every row scored; that is a LogicalTopN / Order, not a plain LogicalLimit over a filter). Opt-in via
// DUCKDB_AI_LIMIT.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class AILimitPushdown {
public:
	explicit AILimitPushdown(Optimizer &optimizer) : optimizer(optimizer) {
	}
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	Optimizer &optimizer;

	//! Depth-first; for each plain constant LIMIT k (offset 0) over row-preserving projections above an
	//! ai_filter tree, push k into the filter's ai_function_with_embed node.
	void Visit(unique_ptr<LogicalOperator> &op);
	//! Ensure the filter has an ai_function_with_embed node (build one from raw ai_filters if needed) and
	//! set its LIMIT to k. Returns true if applied.
	bool ApplyLimit(LogicalOperator &filter_op, int64_t k);
	//! Stage-3 shape (Filter[colref] -> Region[call]): push k into the AI region below the filter -- its
	//! count-weighted Sink early-stop stops after k passing rows, broadcasting the rest false. A scalar call
	//! is converted to the combined node first (only the node path early-stops). Returns true if applied.
	bool ApplyLimitToRegion(LogicalOperator &filter_op, int64_t k);
	//! One combined ai_function_with_embed node over the given AI boolean predicates (ANDed); null if any
	//! predicate is not a convertible AI tree or binding fails.
	unique_ptr<Expression> BuildCombinedNode(const vector<Expression *> &exprs);
	//! A LIMIT k sitting directly above an OUTER join: push k into the AI filter on the join's PRESERVED
	//! side (through row-preserving projections). Sound because every preserved-side row yields >=1 join
	//! output row, so k passers there produce >=k output rows. Inner / semi / anti joins preserve no side
	//! (a passer can produce 0 output rows) and are left untouched. Returns true if applied.
	bool TryPushBelowPreservingJoin(LogicalOperator &join_op, int64_t k);
	//! Cap the AI-bearing side of a CROSS PRODUCT with its own LIMIT k.
	bool TryPushBelowCrossProduct(LogicalOperator &cross_op, int64_t k);
	//! Bind an ai_function_with_embed(tree_str, prompts.., preds.., inputs..) call. Null on bind failure.
	unique_ptr<Expression> BuildNode(vector<unique_ptr<Expression>> args);
};

} // namespace duckdb
