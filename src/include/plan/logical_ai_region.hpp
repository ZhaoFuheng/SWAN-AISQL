//===----------------------------------------------------------------------===//
// plan/logical_ai_region.hpp — the AI Region logical operator (extension operator)
//
// LogicalAIRegion marks the AI region of the plan: it folds its child's output into the
// factorized AI currency (distinct AI-input reps + counts), evaluates a single AI-function call
// once per distinct rep, and expands the (broadcast) result back as one new column bound to
// (region_index, 0). The call is held in expressions[0] so the column-binding resolver rewrites
// its child column refs normally (LogicalExtensionOperator's default resolver does exactly that).
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/planner/operator/logical_extension_operator.hpp"

namespace duckdb {

class LogicalAIRegion : public LogicalExtensionOperator {
public:
	//! Stable identity: opts into exact binding/type verification and keys TryCast matching.
	static const string IDENTIFIER; // "aisql.ai_region"

	LogicalAIRegion(TableIndex region_index, unique_ptr<Expression> eval_expr);

	//! Table index of the single appended result column, bound as (region_index, 0).
	TableIndex region_index;
	//! LIMIT k pushed down from a plain LIMIT above (never ORDER BY ... LIMIT): stop evaluating
	//! distinct inputs once the fan-out counts of PASSING ones sum to >= k. -1 = no limit.
	int64_t limit = -1;
	//! When the LIMIT reached the region through a plain DISTINCT, the child columns the DISTINCT keys on
	//! (positions in the region's child output): the early stop then counts DISTINCT passing tuples over
	//! these columns, not passing rows. Empty = count rows.
	vector<idx_t> limit_distinct_cols;

public:
	//! Downcast helper replacing the fork's enum check: matches on the verification identifier,
	//! safe across shared-library boundaries (no RTTI dependence).
	static LogicalAIRegion *TryCast(LogicalOperator &op);
	//! TryCast's test on a const operator.
	static bool Is(const LogicalOperator &op);

	vector<ColumnBinding> GetColumnBindings() override;
	vector<TableIndex> GetTableIndex() const override;
	string GetName() const override;
	string GetExtensionName() const override;
	optional_ptr<const string> GetTypeBindingVerificationIdentifier() const noexcept override;
	bool SupportSerialization() const override {
		return false;
	}
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;

protected:
	void ResolveTypes() override;
};

} // namespace duckdb
