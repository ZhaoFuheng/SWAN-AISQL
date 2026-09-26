#include "plan/logical_ai_factor_graph.hpp"

#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "exec/physical_ai_factor_graph.hpp"

namespace duckdb {

LogicalAIFactorGraph::LogicalAIFactorGraph(TableIndex result_index, unique_ptr<Expression> node)
    : result_index(result_index) {
	expressions.push_back(std::move(node));
}

LogicalAIFactorGraph *LogicalAIFactorGraph::TryCast(LogicalOperator &op) {
	if (op.type != LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR) {
		return nullptr;
	}
	auto &ext = op.Cast<LogicalExtensionOperator>();
	// Name-keyed like the factor join: a multi-child operator must NOT expose the verification
	// identifier (that opts into the resolver's single-stream traversal).
	if (ext.GetExtensionName() != "aisql" || ext.GetName() != "AI_FACTOR_GRAPH") {
		return nullptr;
	}
	return static_cast<LogicalAIFactorGraph *>(&ext);
}

vector<ColumnBinding> LogicalAIFactorGraph::GetColumnBindings() {
	vector<ColumnBinding> bindings;
	for (auto &child : children) {
		auto child_bindings = child->GetColumnBindings();
		bindings.insert(bindings.end(), child_bindings.begin(), child_bindings.end());
	}
	bindings.emplace_back(result_index, ProjectionIndex(0));
	return bindings;
}

void LogicalAIFactorGraph::ResolveTypes() {
	for (auto &child : children) {
		types.insert(types.end(), child->types.begin(), child->types.end());
	}
	types.push_back(LogicalType::BOOLEAN); // the constant-true survivor column
}

vector<TableIndex> LogicalAIFactorGraph::GetTableIndex() const {
	return vector<TableIndex> {result_index};
}

string LogicalAIFactorGraph::GetName() const {
	return "AI_FACTOR_GRAPH";
}

string LogicalAIFactorGraph::GetExtensionName() const {
	return "aisql";
}

//! Resolve column refs in `expr` against the combined (concatenated-sides) binding list.
static void ResolveAgainst(unique_ptr<Expression> &expr, const vector<ColumnBinding> &combined) {
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto &colref = expr->Cast<BoundColumnRefExpression>();
		for (idx_t i = 0; i < combined.size(); i++) {
			if (combined[i] == colref.Binding()) {
				expr = make_uniq<BoundReferenceExpression>(expr->GetReturnType(), i);
				return;
			}
		}
		throw InternalException("AIFactorGraph: node references a column outside the join inputs");
	}
	ExpressionIterator::EnumerateChildren(*expr,
	                                      [&](unique_ptr<Expression> &child) { ResolveAgainst(child, combined); });
}

void LogicalAIFactorGraph::ResolveColumnBindings(ColumnBindingResolver &res, vector<ColumnBinding> &bindings) {
	// The node's expressions see EVERY side in concatenated layout; the resolver's single-stream
	// traversal cannot express that, so the children resolve normally and the node resolves here.
	vector<ColumnBinding> combined;
	for (auto &child : children) {
		res.VisitOperator(*child);
		auto child_bindings = child->GetColumnBindings();
		combined.insert(combined.end(), child_bindings.begin(), child_bindings.end());
	}
	for (auto &expression : expressions) {
		ResolveAgainst(expression, combined);
	}
	bindings = GetColumnBindings();
}

PhysicalOperator &LogicalAIFactorGraph::CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) {
	D_ASSERT(children.size() >= 2);
	D_ASSERT(expressions.size() == 1);
	auto card = EstimateCardinality(context); // must precede child planning (bind_data moves out)
	vector<reference<PhysicalOperator>> plans;
	vector<idx_t> side_widths;
	for (auto &child : children) {
		auto &plan = planner.CreatePlan(*child);
		side_widths.push_back(plan.GetTypes().size());
		plans.push_back(plan);
	}
	auto &graph = planner.Make<PhysicalAIFactorGraph>(std::move(types), std::move(expressions[0]),
	                                                  std::move(side_widths), limit, existential_side, card);
	for (auto &plan : plans) {
		graph.children.push_back(plan);
	}
	return graph;
}

} // namespace duckdb
