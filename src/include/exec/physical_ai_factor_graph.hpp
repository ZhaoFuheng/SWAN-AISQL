//===----------------------------------------------------------------------===//
// exec/physical_ai_factor_graph.hpp — the n-ary factorized semantic join operator
//
// Every side (child) sinks into its own dictionary of DISTINCT rows + counts (IEJoin-style: k
// sequential build pipelines into one sink, then this operator becomes the source). Evaluation
// is factorized with exact backward pruning:
//   1. unary leaves run over each side's members; a false member is deleted, and with it every
//      tuple that would have contained it (pruning in member currency, not tuple currency);
//   2. edges (binary leaves) run over the SURVIVING pair domains, cheapest domain first; after
//      each edge, deletions cascade to fixed point (a member with no passing partner on an
//      evaluated incident edge dies, shrinking every remaining domain);
//   3. only surviving tuples are enumerated (DFS over the edge graph), expanded by count
//      products. The k-way cross product never exists in row currency.
//===----------------------------------------------------------------------===//
#pragma once

#include "ai_dedup.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"

#include <unordered_map>

namespace duckdb {
class BoundFunctionExpression;

class PhysicalAIFactorGraph : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

	PhysicalAIFactorGraph(PhysicalPlan &physical_plan, vector<LogicalType> types, unique_ptr<Expression> node,
	                      vector<idx_t> side_widths, int64_t limit, idx_t existential_side,
	                      idx_t estimated_cardinality);

	//! The folded AI node over the concatenated-sides layout: a conjunction of factors, each a one- or
	//! two-side Boolean tree over its leaves.
	unique_ptr<Expression> node;
	//! Column count per side; prefix sums give each side's offset in the concatenated layout.
	vector<idx_t> side_widths;
	vector<idx_t> side_offsets;
	//! The factors (from AIFactorDecompose in the constructor).
	vector<AIFactor> factors;
	//! Unary leaf ids per side, and the side's Boolean tree over them (the AND of its unary factors,
	//! serialized over positions in `unary_leaves[s]`).
	vector<vector<idx_t>> unary_leaves;
	vector<string> unary_trees;
	//! Edges: (side s, side t, the leaf ids of the factors connecting them, their AND as a tree over
	//! positions in `leaf_ids`), s < t.
	struct Edge {
		idx_t s;
		idx_t t;
		vector<idx_t> leaf_ids;
		string tree;
	};
	vector<Edge> edges;
	//! LIMIT k pushed into the evaluation (-1 = none): the lazy/adaptive scheduler stops
	//! dispatching once k output rows are confirmed.
	int64_t limit;
	//! Existentially-consumed side (INVALID = none): see LogicalAIFactorGraph::existential_side.
	idx_t existential_side;
	//! Pipeline-source operator -> side id, filled in BuildPipelines: each build pipeline's local
	//! sink state resolves its side from its pipeline's source, independent of scheduling order.
	std::unordered_map<const PhysicalOperator *, idx_t> source_to_side;

public:
	string GetName() const override {
		return "AI_FACTOR_GRAPH";
	}
	InsertionOrderPreservingMap<string> ParamsToString() const override;

	// Sink interface (every child builds its side dictionary; sides finish in child order k-1..0)
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

	// Source interface (first GetData runs the factorized evaluation, then paginates survivors)
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;
	bool IsSource() const override {
		return true;
	}
	bool ParallelSource() const override {
		return false;
	}

	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;
	vector<const_reference<PhysicalOperator>> GetSources() const override {
		// sink + source over k children: this operator is the source of its own pipeline
		return vector<const_reference<PhysicalOperator>> {*this};
	}
};

} // namespace duckdb
