//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/filter/physical_ai_dedup.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

//! PhysicalAIRegion is the AI region of the plan -- the boundary where the relational currency (exploded
//! DataChunks) folds into the AI currency (a factorized AISQLMapData: distinct AI-input rep rows + a __count
//! multiplicity). It folds its child's output into that factorized form, fires the AI call once per distinct
//! rep, then expands the result back to every row as a new column. The rewrite places it above a join so the
//! calls run once per distinct input over the whole join output instead of fragmented across the 2048-row
//! chunks the join streams up.
//!
//! Streaming by default (ai_debug_region_blocking=true falls back to one global evaluation in Finalize). A
//! folded NODE is held per leaf (exec/ai_leaf_region.hpp: |A| + |B| reps, per-row leaf order, leaf waves in
//! prompt currency); a plain SCALAR call folds into an AISQLMapChunk on its key columns and fires a WAVE once
//! >= ai_debug_wave_size (default 5 x ai_concurrency) new distinct reps have accumulated. In practice a wave is one input chunk's worth of
//! new reps, because a wave drains everything pending and every chunk clears the floor.
//!
//! A wave's LLM batch is only as wide as the DISTINCT PROMPTS its reps carry, not its rep count: above a
//! join that repeats one side's prompt across a whole chunk, a wave can collapse to a single call. So up to
//! ai_debug_wave_overlap waves (default 8) run concurrently -- prepared on the Sink thread (the map is
//! single-owner), evaluated on a background thread, applied back on the Sink thread, drained in Combine --
//! which keeps the request pool fed without speculating on any call the tree would not have made. The
//! process-wide chat gate still caps requests in flight at ai_concurrency. Under a pushed LIMIT waves stay
//! inline, since there the floor's job is to stop early (Sink returns FINISHED once the passing outputs
//! reach k). Single-owner sink.
class PhysicalAIRegion : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

public:
	PhysicalAIRegion(PhysicalPlan &physical_plan, vector<LogicalType> types, unique_ptr<Expression> eval_call,
	                idx_t estimated_cardinality, int64_t limit = -1);

	//! The AI call (a BoundFunctionExpression) evaluated once per distinct input and broadcast to all rows.
	unique_ptr<Expression> eval_call;
	//! LIMIT k (see LogicalAIRegion::limit); -1 = evaluate every distinct input.
	int64_t limit;

public:
	string GetName() const override {
		return "AI_REGION";
	}
	InsertionOrderPreservingMap<string> ParamsToString() const override;

	// Source interface
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;
	bool IsSource() const override {
		return true;
	}
	bool ParallelSource() const override {
		return false;
	}

	// Sink interface
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;
	bool IsSink() const override {
		return true;
	}
	//! Single-owner sink: the streaming dedup map + wave firing must be driven by one thread, and a
	//! non-parallel sink is what lets Sink() return FINISHED to early-terminate the feeding pipeline.
	bool ParallelSink() const override {
		return false;
	}
};

} // namespace duckdb
