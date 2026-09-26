//===----------------------------------------------------------------------===//
// plan/logical_ai_factor_graph.hpp — the n-ary factorized semantic join
//
// Generalizes the 2-way factor join to a FACTOR GRAPH over k join sides: one dictionary of
// distinct rows + counts per side, the conjunctive folded AI node split by column reads into
// unary leaves (attached to one side) and binary leaves (edges between two sides). Evaluation
// prunes factorized: unary leaves run over each side's distinct members first (a false member is
// deleted before any tuple containing it exists), then edges run over the SURVIVING pair domains
// in ascending-size order with backward cascade (a member with no passing partner on an incident
// edge dies, shrinking the edges not yet evaluated). Only surviving tuples are enumerated,
// expanded by count products — the k-way cross product is never materialized.
//
// The rewrite replaces  Filter(#r) -> AIRegion(folded node) -> CrossProduct chain(S0..Sk-1)
// with this operator, exposing the same bindings the filter saw (combined side bindings +
// (result_index, 0) as a constant-true BOOLEAN column).
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/planner/operator/logical_extension_operator.hpp"

namespace duckdb {

class LogicalAIFactorGraph : public LogicalExtensionOperator {
public:
	LogicalAIFactorGraph(TableIndex result_index, unique_ptr<Expression> node);

	//! Table index of the appended constant-true result column (the region binding it replaces).
	TableIndex result_index;
	//! LIMIT k pushed into the evaluation (AILimitPushdown): stop once k output rows are
	//! CONFIRMED (count products included); unevaluated tuples are simply not emitted. -1 = none.
	int64_t limit = -1;
	//! Side consumed EXISTENTIALLY (a MARK join above needs only a boolean per member of this
	//! side): each member completes at its FIRST confirmed tuple, and downstream work flows only
	//! for unsatisfied members. INVALID = none.
	idx_t existential_side = DConstants::INVALID_INDEX;

public:
	static LogicalAIFactorGraph *TryCast(LogicalOperator &op);

	vector<ColumnBinding> GetColumnBindings() override;
	vector<TableIndex> GetTableIndex() const override;
	string GetName() const override;
	string GetExtensionName() const override;
	bool SupportSerialization() const override {
		return false;
	}
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;
	void ResolveColumnBindings(ColumnBindingResolver &res, vector<ColumnBinding> &bindings) override;

protected:
	void ResolveTypes() override;
};

} // namespace duckdb
