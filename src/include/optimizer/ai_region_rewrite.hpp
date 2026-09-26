//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/ai_dedup_rewrite.hpp
//
// Inserts LogicalAIRegion ops so AI calls consume the factorized AISQLMapData currency: fold the input to
// distinct reps, evaluate ONCE per rep (full concurrency at the cardinality floor, no per-chunk barriers),
// fan the result back out. Two placements: (1) above a JOIN cluster -- the fan-out dedup that removes the
// pull-up penalty; (2) STAGE 3, above ANY child (a base table can hold many duplicates, and per-chunk scalar
// batches under-saturate concurrency) -- gated call-safe (all-AI filters only, no lazy CASE/conjunction
// positions, LIMIT defers). Opt-out via DUCKDB_AI_DEDUP=off (all) / DUCKDB_AI_SCAN_REGION=off (scan path).
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class AIRegionRewrite {
public:
	explicit AIRegionRewrite(Optimizer &optimizer) : optimizer(optimizer) {
	}
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	Optimizer &optimizer;
	//! Whether to also dedup above fan-out joins (INNER / cross product), which CHANGES the call count by
	//! evaluating each distinct input once -- opt-in via DUCKDB_AI_DEDUP. When false (default), only the
	//! concurrency-preserving SEMI-reducer buffering runs (no call-count or result change).
	bool allow_fanout_dedup = false;
	bool scan_regions = false;

	//! Depth-first; hoist AI calls out of each join-adjacent Filter/Projection into LogicalAIRegion ops.
	//! `limit_k` (>= 0) = the enclosing constant LIMIT k reached through row-preserving projections; it is
	//! pushed into a hoisted ai_function_with_embed node's LogicalAIRegion for a count-weighted early-stop
	//! (scalar AI calls under a limit are left in place to stream). -1 = no enclosing limit.
	void Rewrite(unique_ptr<LogicalOperator> &op, int64_t limit_k);
	//! Recursively hoist every dedupable AI scalar call in `expr` (top-level OR nested, e.g. the
	//! ai_classify(x) inside ai_classify(x)='a') into a LogicalAIRegion stacked onto `child_slot`, replacing
	//! each call with a colref to its deduplicated result column. `limit_k` per Rewrite. `lazy_guard` (the
	//! Stage-3 scan path) skips lazily-evaluated positions -- CASE beyond the first WHEN condition and
	//! conjunction arms -- so factorization never evaluates inputs native execution would have short-circuited
	//! away (never MORE calls). Returns true if anything was hoisted.
	bool HoistAICalls(unique_ptr<Expression> &expr, unique_ptr<LogicalOperator> &child_slot, int64_t limit_k,
	                  bool lazy_guard = false);
};

} // namespace duckdb
