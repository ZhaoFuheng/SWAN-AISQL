#include "optimizer/ai_topn_pushdown.hpp"

#include "ai_settings.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"

namespace duckdb {

static bool ExprHasAICall(const Expression &e) {
	if (e.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
		const auto &name = e.Cast<BoundFunctionExpression>().Function().GetName();
		if (name.StartsWith("ai_") || name.StartsWith("speculative_ai_")) {
			return true;
		}
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(e, [&](const Expression &child) {
		if (!found && ExprHasAICall(child)) {
			found = true;
		}
	});
	return found;
}

//! Replace every reference to the projection's output by the projection's expression. Fails (ok = false) when
//! a key reads an AI result or a volatile expression, which must not be evaluated twice.
static void SubstituteProjection(unique_ptr<Expression> &e, const LogicalProjection &proj, bool &ok) {
	if (!ok) {
		return;
	}
	if (e->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		const auto &binding = e->Cast<BoundColumnRefExpression>().Binding();
		if (binding.table_index != proj.table_index) {
			ok = false; // not the projection's column: the key reads something this rewrite cannot see
			return;
		}
		const auto &source = proj.expressions[binding.column_index.GetIndex()];
		if (ExprHasAICall(*source) || source->IsVolatile()) {
			ok = false;
			return;
		}
		e = source->Copy();
		return;
	}
	ExpressionIterator::EnumerateChildren(
	    *e, [&](unique_ptr<Expression> &child) { SubstituteProjection(child, proj, ok); });
}

void AITopNPushdown::Visit(unique_ptr<LogicalOperator> &op) {
	for (auto &child : op->children) {
		Visit(child);
	}
	if (op->type != LogicalOperatorType::LOGICAL_TOP_N || op->children.size() != 1 ||
	    op->children[0]->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return;
	}
	auto &proj = op->children[0]->Cast<LogicalProjection>();
	bool has_ai = false;
	for (auto &e : proj.expressions) {
		has_ai = has_ai || ExprHasAICall(*e);
	}
	if (!has_ai || proj.children.size() != 1) {
		return;
	}
	auto &top_n = op->Cast<LogicalTopN>();
	vector<unique_ptr<Expression>> keys;
	bool ok = true;
	for (auto &order : top_n.orders) {
		auto key = order.expression->Copy();
		SubstituteProjection(key, proj, ok);
		if (!ok) {
			return;
		}
		keys.push_back(std::move(key));
	}
	for (idx_t i = 0; i < keys.size(); i++) {
		top_n.orders[i].expression = std::move(keys[i]);
	}
	// TOP_N[P[child]] -> P[TOP_N[child]]
	auto projection = std::move(op->children[0]);
	op->children[0] = std::move(projection->children[0]);
	op->ResolveOperatorTypes();
	projection->children[0] = std::move(op);
	op = std::move(projection);
}

unique_ptr<LogicalOperator> AITopNPushdown::Optimize(unique_ptr<LogicalOperator> op) {
	if (AIBoolSetting(optimizer.context, "ai_limit", true)) {
		Visit(op);
	}
	return op;
}

} // namespace duckdb
