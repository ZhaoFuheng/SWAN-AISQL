//===----------------------------------------------------------------------===//
// exec/physical_ai_factor_join.hpp — the factorized semantic join operator
//
// Build side (children[1]) sinks into a dictionary of DISTINCT full rows + counts. Probe side
// (children[0]) streams: each chunk's previously-unseen predicate-key reps are evaluated against
// every build rep as factorized pair batches (the LLM client dedups repeated prompts), producing
// a passset per probe rep; each input row then emits (row x passing build reps x build counts),
// paginated across Execute calls. The cross product is never materialized: only survivors are.
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

class PhysicalAIFactorJoin : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

	PhysicalAIFactorJoin(PhysicalPlan &physical_plan, vector<LogicalType> types, unique_ptr<Expression> predicate,
	                     idx_t left_width, idx_t estimated_cardinality);

	//! The AI predicate over the cross layout [left cols..., right cols...].
	unique_ptr<Expression> predicate;
	//! Number of probe-side (left) columns in the cross layout.
	idx_t left_width;
	//! Cross-layout column indices the predicate references on the probe side (its dedup key).
	vector<idx_t> left_key_cols;

public:
	string GetName() const override {
		return "AI_FACTOR_JOIN";
	}
	InsertionOrderPreservingMap<string> ParamsToString() const override;

	// Operator interface (probe side streams through)
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;

	// Sink interface (build side)
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;
	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return false;
	}
	bool ParallelOperator() const override {
		return false;
	}

	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;
	vector<const_reference<PhysicalOperator>> GetSources() const override;
};

} // namespace duckdb
