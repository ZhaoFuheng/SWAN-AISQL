//===----------------------------------------------------------------------===//
// plan/logical_ai_factor_join.hpp — the factorized semantic join (Concept 3)
//
// A semantic join is a cartesian product filtered by an AI predicate. Instead of materializing
// the cross product and folding it in a region (rows = |L| x |R| flow through the plan), this
// operator keeps BOTH sides factorized: the build side folds to its distinct rows + counts, the
// probe side streams and folds to distinct predicate inputs; the AI predicate is evaluated once
// per distinct (probe-rep x build-rep) pair, and ONLY PASSING pairs are emitted, expanded by
// their multiplicities. The intermediate cross product never exists.
//
// The rewrite replaces  Filter(#r) -> AIRegion(pred) -> CrossProduct(L, R)  with this operator,
// which exposes the SAME bindings the filter did (cross bindings + (result_index, 0) as a
// constant-true BOOLEAN column), so every upstream reference stays valid.
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/planner/operator/logical_extension_operator.hpp"

namespace duckdb {

class LogicalAIFactorJoin : public LogicalExtensionOperator {
public:
	static const string IDENTIFIER; // "aisql.ai_factor_join"

	LogicalAIFactorJoin(TableIndex result_index, unique_ptr<Expression> predicate);

	//! Table index of the appended constant-true result column (the region binding it replaces).
	TableIndex result_index;

public:
	static LogicalAIFactorJoin *TryCast(LogicalOperator &op);

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
