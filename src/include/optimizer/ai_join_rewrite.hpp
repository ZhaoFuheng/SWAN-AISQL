//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/ai_group_join_rewrite.hpp
//
// Stage 2 group-join (opt-in via DUCKDB_AI_GROUP_JOIN). The Stage-1 AIRegionRewrite places a LogicalAIRegion
// ABOVE a join, so the join materializes its full N*M fan-out and the region SINKS all of it just to dedup the
// AI call down to the distinct inputs. When the region's AI call reads columns from only ONE side S of the join
// (an INNER join or cross product), this pass PUSHES the region down through the join onto S:
//     Region[ Join(L, R) ]   (AI key columns ⊆ L)   ->   Join( Region[L], R )
// The region now folds/evaluates the AI over S's rows (already reduced to the surviving join keys by the
// Yannakakis semi-join reducer, which runs earlier and is purely relational -- no AI in the reduction), then the
// join fans the single appended result column out to every output row. Same LLM calls as Stage 1 (both dedup to
// the surviving distinct keys), identical results, but the region buffers |S| rows instead of the whole fan-out.
// Multi-side keys (the AI call reads both sides) keep the Stage-1 region above the join.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class AIJoinRewrite {
public:
	explicit AIJoinRewrite(Optimizer &optimizer) : optimizer(optimizer) {
	}
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	Optimizer &optimizer;
	//! ai_join_factorize mode: 'pushdown' or 'factor' (off early-outs in Optimize).
	bool factor_mode = false;

	//! Depth-first; after processing children, try to push a single-side-key LogicalAIRegion below its join.
	//! `root` is the plan root, used to rebind consumers of the region's result column after a push.
	void Rewrite(unique_ptr<LogicalOperator> &op, unique_ptr<LogicalOperator> &root);
	//! If `op` is a LogicalAIRegion above an INNER join / cross product (directly or through one projection)
	//! whose AI call reads columns from exactly one side, push the region onto that side (the projection's
	//! expressions are inlined into the call; consumers are rebound through the projection). No-op otherwise.
	void TryPushBelowJoin(unique_ptr<LogicalOperator> &op, unique_ptr<LogicalOperator> &root);
	//! Factor mode: replace Filter(#region) -> AIRegion(pred) -> CrossProduct(L, R) (a pulled-up
	//! two-side AI join condition) with a LogicalAIFactorJoin -- the cross product is never
	//! materialized; only passing pairs are emitted. No-op unless the pattern matches exactly.
	bool TryFactorJoin(unique_ptr<LogicalOperator> &op);
	bool TryFactorGraph(unique_ptr<LogicalOperator> &op);
	void TryMarkExistential(LogicalOperator &op);
	//! Depth-first sweep for duplicate-insensitive consumers (DISTINCT / min/max/DISTINCT-agg GROUP BY) whose
	//! whole input derives from ONE side of an INNER join below: converts the join to (RIGHT_)SEMI so the
	//! fan-out is never produced and the expand disappears (each kept-side row flows exactly once).
	void SemiConvertSweep(LogicalOperator &op);
	bool TrySemiConvertForConsumer(LogicalOperator &consumer);
	//! Whether SemiConvertSweep changed a join (the plan's types then need one re-resolve).
	bool semi_converted = false;
};

} // namespace duckdb
