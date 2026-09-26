#include "plan/logical_ai_factor_join.hpp"

#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "exec/physical_ai_factor_join.hpp"

namespace duckdb {

const string LogicalAIFactorJoin::IDENTIFIER = "aisql.ai_factor_join";

LogicalAIFactorJoin::LogicalAIFactorJoin(TableIndex result_index, unique_ptr<Expression> predicate)
    : result_index(result_index) {
	expressions.push_back(std::move(predicate));
}

LogicalAIFactorJoin *LogicalAIFactorJoin::TryCast(LogicalOperator &op) {
	if (op.type != LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR) {
		return nullptr;
	}
	auto &ext = op.Cast<LogicalExtensionOperator>();
	// Keyed on extension + operator name: this operator does NOT expose the verification
	// identifier, because that opts into the resolver's single-stream standard traversal, which
	// cannot express a two-child (join-layout) expression context.
	if (ext.GetExtensionName() != "aisql" || ext.GetName() != "AI_FACTOR_JOIN") {
		return nullptr;
	}
	return static_cast<LogicalAIFactorJoin *>(&ext);
}

vector<ColumnBinding> LogicalAIFactorJoin::GetColumnBindings() {
	auto bindings = children[0]->GetColumnBindings();
	auto right = children[1]->GetColumnBindings();
	bindings.insert(bindings.end(), right.begin(), right.end());
	bindings.emplace_back(result_index, ProjectionIndex(0));
	return bindings;
}

void LogicalAIFactorJoin::ResolveTypes() {
	types.insert(types.end(), children[0]->types.begin(), children[0]->types.end());
	types.insert(types.end(), children[1]->types.begin(), children[1]->types.end());
	types.push_back(LogicalType::BOOLEAN); // the constant-true survivor column
}

vector<TableIndex> LogicalAIFactorJoin::GetTableIndex() const {
	return vector<TableIndex> {result_index};
}

string LogicalAIFactorJoin::GetName() const {
	return "AI_FACTOR_JOIN";
}

string LogicalAIFactorJoin::GetExtensionName() const {
	return "aisql";
}

//! Resolve column refs in `expr` against the combined (cross-layout) binding list.
static void ResolveAgainst(unique_ptr<Expression> &expr, const vector<ColumnBinding> &combined,
                           const vector<LogicalType> &types) {
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto &colref = expr->Cast<BoundColumnRefExpression>();
		for (idx_t i = 0; i < combined.size(); i++) {
			if (combined[i] == colref.Binding()) {
				expr = make_uniq<BoundReferenceExpression>(expr->GetReturnType(), i);
				return;
			}
		}
		throw InternalException("AIFactorJoin: predicate references a column outside the join inputs");
	}
	ExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<Expression> &child) { ResolveAgainst(child, combined, types); });
}

void LogicalAIFactorJoin::ResolveColumnBindings(ColumnBindingResolver &res, vector<ColumnBinding> &bindings) {
	// Two-child operator: the predicate sees BOTH sides in cross layout (left columns first),
	// like a join. The resolver's single-stream traversal cannot express that, so the children
	// resolve normally and the predicate is resolved here against the combined layout.
	res.VisitOperator(*children[0]);
	res.VisitOperator(*children[1]);
	auto combined = children[0]->GetColumnBindings();
	auto right = children[1]->GetColumnBindings();
	combined.insert(combined.end(), right.begin(), right.end());
	for (auto &expression : expressions) {
		ResolveAgainst(expression, combined, types);
	}
	bindings = GetColumnBindings();
}

PhysicalOperator &LogicalAIFactorJoin::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	D_ASSERT(children.size() == 2);
	D_ASSERT(expressions.size() == 1);
	auto card = EstimateCardinality(context); // must precede child planning (bind_data moves out)
	auto &left = planner.CreatePlan(*children[0]);
	auto &right = planner.CreatePlan(*children[1]);
	const idx_t left_width = left.GetTypes().size();
	auto &join = planner.Make<PhysicalAIFactorJoin>(std::move(types), std::move(expressions[0]), left_width, card);
	join.children.push_back(left);
	join.children.push_back(right);
	return join;
}

} // namespace duckdb
