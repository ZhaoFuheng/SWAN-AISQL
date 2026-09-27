#include "plan/logical_ai_region.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "exec/physical_ai_region.hpp"

namespace duckdb {

const string LogicalAIRegion::IDENTIFIER = "aisql.ai_region";

LogicalAIRegion::LogicalAIRegion(TableIndex region_index, unique_ptr<Expression> eval_expr)
    : region_index(region_index) {
	expressions.push_back(std::move(eval_expr));
}

bool LogicalAIRegion::Is(const LogicalOperator &op) {
	if (op.type != LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR) {
		return false;
	}
	auto id = op.Cast<LogicalExtensionOperator>().GetTypeBindingVerificationIdentifier();
	return id && *id == IDENTIFIER;
}

LogicalAIRegion *LogicalAIRegion::TryCast(LogicalOperator &op) {
	if (op.type != LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR) {
		return nullptr;
	}
	auto &ext = op.Cast<LogicalExtensionOperator>();
	auto id = ext.GetTypeBindingVerificationIdentifier();
	if (!id || *id != IDENTIFIER) {
		return nullptr;
	}
	return static_cast<LogicalAIRegion *>(&ext);
}

vector<ColumnBinding> LogicalAIRegion::GetColumnBindings() {
	auto child_bindings = children[0]->GetColumnBindings();
	child_bindings.emplace_back(region_index, ProjectionIndex(0));
	return child_bindings;
}

void LogicalAIRegion::ResolveTypes() {
	types.insert(types.end(), children[0]->types.begin(), children[0]->types.end());
	types.push_back(expressions[0]->GetReturnType());
}

vector<TableIndex> LogicalAIRegion::GetTableIndex() const {
	return vector<TableIndex> {region_index};
}

string LogicalAIRegion::GetName() const {
	return "AI_REGION";
}

string LogicalAIRegion::GetExtensionName() const {
	return "aisql";
}

optional_ptr<const string> LogicalAIRegion::GetTypeBindingVerificationIdentifier() const noexcept {
	return &IDENTIFIER;
}

PhysicalOperator &LogicalAIRegion::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	D_ASSERT(children.size() == 1);
	D_ASSERT(expressions.size() == 1);
	// Estimate BEFORE planning the child: physical planning moves each LogicalGet's bind_data
	// into the physical operator, and estimating afterwards recurses into gutted children
	// (nullptr bind_data -> crash) whenever no earlier pass cached the estimate.
	auto card = EstimateCardinality(context);
	auto &child = planner.CreatePlan(*children[0]);
	auto &region = planner.Make<PhysicalAIRegion>(std::move(types), std::move(expressions[0]), card, limit);
	region.children.push_back(child);
	return region;
}

} // namespace duckdb
