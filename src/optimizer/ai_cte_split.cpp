#include "optimizer/ai_cte_split.hpp"

#include "duckdb/common/projection_index.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"

#include <functional>

namespace duckdb {

//! Row-wise AI functions: a predicate containing one costs an LLM call per distinct input, which
//! is the only reason this pass exists.
static bool IsAIFunctionName(const Identifier &name) {
	return name == "ai_filter" || name == "ai_classify" || name == "ai_score" || name == "ai_complete" ||
	       name == "ai_function_with_embed";
}

static bool ExpressionHasAIFunction(const Expression &expr) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION &&
	    IsAIFunctionName(expr.Cast<BoundFunctionExpression>().Function().GetName())) {
		return true;
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!found && ExpressionHasAIFunction(child)) {
			found = true;
		}
	});
	return found;
}

//! The body's single AI-bearing filter. More than one and the pass declines, rather than invent an
//! order between them -- keeping the rule to one shape is the point.
static optional_ptr<LogicalFilter> FindSoleAIFilter(LogicalOperator &op) {
	optional_ptr<LogicalFilter> found;
	bool ambiguous = false;
	std::function<void(LogicalOperator &)> walk = [&](LogicalOperator &node) {
		if (node.type == LogicalOperatorType::LOGICAL_FILTER) {
			for (auto &expr : node.expressions) {
				if (ExpressionHasAIFunction(*expr)) {
					if (found) {
						ambiguous = true;
					}
					found = &node.Cast<LogicalFilter>();
					break;
				}
			}
		}
		for (auto &child : node.children) {
			walk(*child);
		}
	};
	walk(op);
	return ambiguous ? optional_ptr<LogicalFilter>() : found;
}

//! Point everything that read the relational prefix's columns at the CTE reference now standing in
//! for it. Positional: the reference exposes the prefix's columns in order.
static void RepointToPrefixRef(LogicalOperator &op, const vector<ColumnBinding> &prefix_bindings,
                               TableIndex ref_index) {
	std::function<void(Expression &)> rewrite = [&](Expression &expr) {
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
			auto &colref = expr.Cast<BoundColumnRefExpression>();
			for (idx_t i = 0; i < prefix_bindings.size(); i++) {
				if (prefix_bindings[i] == colref.Binding()) {
					colref.BindingMutable() = ColumnBinding(ref_index, ProjectionIndex(i));
					return;
				}
			}
			return;
		}
		ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { rewrite(child); });
	};
	std::function<void(LogicalOperator &)> walk = [&](LogicalOperator &node) {
		for (auto &expr : node.expressions) {
			rewrite(*expr);
		}
		for (auto &child : node.children) {
			walk(*child);
		}
	};
	walk(op);
}

//! How many times this subtree reads the CTE at `cte_index`.
static idx_t CountCTEReads(const LogicalOperator &op, TableIndex cte_index) {
	idx_t n = op.type == LogicalOperatorType::LOGICAL_CTE_REF && op.Cast<LogicalCTERef>().cte_index == cte_index;
	for (auto &child : op.children) {
		n += CountCTEReads(*child, cte_index);
	}
	return n;
}

//! Split one CTE at its semantic boundary. The RELATIONAL prefix below the AI predicate becomes an
//! outer materialized CTE that runs once; what remains -- the predicate over a reference to that
//! prefix -- is marked never-materialize, so DuckDB's inliner substitutes it into each reference.
//!
//! Each reference therefore holds a genuinely INLINED subtree in its own scope, which is what lets
//! the existing machinery work unchanged: filter pushdown pushes that reference's own predicates
//! below the AI predicate, and the semi-join reducer splices its SEMI below it, exactly as for a
//! fully inlined body. The only difference is that the leaf underneath is a scan of a prefix
//! computed ONCE, instead of the whole relational subplan re-executed per reference.
static bool SplitOne(Binder &binder, unique_ptr<LogicalOperator> &op) {
	auto &cte = op->Cast<LogicalMaterializedCTE>();
	if (cte.children.size() != 2 || cte.materialize == CTEMaterialize::CTE_MATERIALIZE_NEVER) {
		return false;
	}
	auto ai_filter = FindSoleAIFilter(*cte.children[0]);
	if (!ai_filter || ai_filter->children.size() != 1) {
		return false;
	}
	// A prefix that reads the CTE cannot be hoisted above the CTE that defines it.
	if (CountCTEReads(*ai_filter->children[0], cte.table_index) > 0) {
		return false;
	}
	// The trade is a materialization barrier in exchange for doing the relational work once. With
	// a SINGLE reference there is no work to share, so the barrier is pure loss: the semi-join
	// reducer translates its keys down through projections to the scan, and it cannot translate
	// through a CTE scan (computed_key_reduction: 10 calls -> 200). Split only shared bodies.
	if (CountCTEReads(*cte.children[1], cte.table_index) < 2) {
		return false;
	}
	ai_filter->children[0]->ResolveOperatorTypes();
	auto prefix_bindings = ai_filter->children[0]->GetColumnBindings();
	auto prefix_types = ai_filter->children[0]->types;
	if (prefix_types.empty() || prefix_bindings.size() != prefix_types.size()) {
		return false;
	}

	const auto prefix_cte_index = binder.GenerateTableIndex();
	const auto prefix_ref_index = binder.GenerateTableIndex();
	vector<Identifier> colnames;
	for (idx_t i = 0; i < prefix_types.size(); i++) {
		colnames.push_back(Identifier("c" + std::to_string(i)));
	}
	auto relational = std::move(ai_filter->children[0]);
	ai_filter->children[0] =
	    make_uniq<LogicalCTERef>(prefix_ref_index, prefix_cte_index, prefix_types, std::move(colnames));
	RepointToPrefixRef(*cte.children[0], prefix_bindings, prefix_ref_index);
	cte.children[0]->ResolveOperatorTypes();

	// what remains of the original CTE is thin -- the predicate over a prefix scan -- so copying it
	// per reference is cheap. That is the whole trade this pass makes.
	cte.materialize = CTEMaterialize::CTE_MATERIALIZE_NEVER;

	auto outer =
	    make_uniq<LogicalMaterializedCTE>(Identifier("__ai_prefix"), prefix_cte_index, prefix_types.size(),
	                                      std::move(relational), std::move(op), CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
	outer->ResolveOperatorTypes();
	op = std::move(outer);
	return true;
}

static bool SplitTree(Binder &binder, unique_ptr<LogicalOperator> &op) {
	bool changed = false;
	if (op->type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
		if (SplitOne(binder, op)) {
			changed = true;
		}
	}
	for (auto &child : op->children) {
		if (SplitTree(binder, child)) {
			changed = true;
		}
	}
	return changed;
}

bool AISplitCTEsAtSemanticBoundary(Binder &binder, unique_ptr<LogicalOperator> &plan) {
	return SplitTree(binder, plan);
}

} // namespace duckdb
