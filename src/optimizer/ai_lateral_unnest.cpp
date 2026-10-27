#include "optimizer/ai_lateral_unnest.hpp"
#include "optimizer/semi_join_reducer.hpp"

#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_unnest_expression.hpp"
#include "duckdb/planner/expression_binder.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_unnest.hpp"

#include <functional>
#include <unordered_map>

namespace duckdb {

static bool IsCTERef(const LogicalOperator &op, TableIndex cte_index) {
	return op.type == LogicalOperatorType::LOGICAL_CTE_REF && op.Cast<LogicalCTERef>().cte_index == cte_index;
}

static idx_t CountCTERefs(const LogicalOperator &op, TableIndex cte_index) {
	idx_t n = IsCTERef(op, cte_index) ? 1 : 0;
	for (auto &child : op.children) {
		n += CountCTERefs(*child, cte_index);
	}
	return n;
}

static void CollectCTEBodies(LogicalOperator &op, std::unordered_map<idx_t, LogicalOperator *> &bodies) {
	if (op.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE && !op.children.empty()) {
		bodies[op.Cast<LogicalMaterializedCTE>().table_index.index] = op.children[0].get();
	}
	for (auto &child : op.children) {
		CollectCTEBodies(*child, bodies);
	}
}

static optional_ptr<const BoundColumnRefExpression> AsColumnRef(const Expression &expr) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return nullptr;
	}
	auto &colref = expr.Cast<BoundColumnRefExpression>();
	return colref.Depth() == 0 ? &colref : nullptr;
}

//! The column of the domain CTE that a dedup group reads: the group names either the column of the domain
//! reference below the aggregate, or the input of the domain body's top projection that produces it.
static optional_idx DomainColumnOfGroup(const Expression &group, const LogicalCTERef &domain_ref,
                                        LogicalOperator &domain_body) {
	auto colref = AsColumnRef(group);
	if (!colref) {
		return optional_idx();
	}
	const auto binding = colref->Binding();
	if (binding.table_index == domain_ref.table_index) {
		return binding.column_index.GetIndex();
	}
	if (domain_body.type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return optional_idx();
	}
	auto &proj = domain_body.Cast<LogicalProjection>();
	for (idx_t k = 0; k < proj.expressions.size(); k++) {
		auto out = AsColumnRef(*proj.expressions[k]);
		if (out && out->Binding() == binding) {
			return k;
		}
	}
	return optional_idx();
}

//! Rewrite one decorrelated lateral unnest rooted at the dedup CTE `dedup_slot`:
//!
//!   CTE DD := GROUP BY g over D                          (no aggregates)
//!     ... PROJECTION* <- INNER JOIN on NOT DISTINCT FROM
//!                          |- CTE_SCAN D                 (the domain side, L)
//!                          '- GET unnest <- PROJECTION P <- CTE_SCAN DD
//!
//! becomes ... PROJECTION* <- UNNEST(f(L)) <- CTE_SCAN D, with DD gone. Declines on anything else.
static bool RewriteOne(Binder &binder, unique_ptr<LogicalOperator> &root, unique_ptr<LogicalOperator> &dedup_slot,
                       const std::unordered_map<idx_t, LogicalOperator *> &bodies) {
	if (dedup_slot->type != LogicalOperatorType::LOGICAL_MATERIALIZED_CTE || dedup_slot->children.size() != 2) {
		return false;
	}
	auto &dedup_cte = dedup_slot->Cast<LogicalMaterializedCTE>();
	auto &dedup_body = *dedup_cte.children[0];
	if (dedup_body.type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY || dedup_body.children.size() != 1 ||
	    dedup_body.children[0]->type != LogicalOperatorType::LOGICAL_CTE_REF) {
		return false;
	}
	auto &dedup = dedup_body.Cast<LogicalAggregate>();
	if (!dedup.expressions.empty() || dedup.grouping_sets.size() > 1 || dedup.groups.empty()) {
		return false;
	}
	auto &domain_ref = dedup.children[0]->Cast<LogicalCTERef>();
	const auto domain_index = domain_ref.cte_index;
	auto body_it = bodies.find(domain_index.index);
	if (body_it == bodies.end()) {
		return false;
	}
	if (CountCTERefs(*root, dedup_cte.table_index) != 1) {
		return false;
	}

	// down the main query's projections to the join
	unique_ptr<LogicalOperator> *join_slot = &dedup_cte.children[1];
	while ((*join_slot)->type == LogicalOperatorType::LOGICAL_PROJECTION && (*join_slot)->children.size() == 1) {
		join_slot = &(*join_slot)->children[0];
	}
	if ((*join_slot)->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return false;
	}
	auto &join = (*join_slot)->Cast<LogicalComparisonJoin>();
	if (join.join_type != JoinType::INNER || join.children.size() != 2 || join.conditions.empty()) {
		return false;
	}
	idx_t domain_side;
	if (IsCTERef(*join.children[0], domain_index)) {
		domain_side = 0;
	} else if (IsCTERef(*join.children[1], domain_index)) {
		domain_side = 1;
	} else {
		return false;
	}
	auto &domain_scan = join.children[domain_side]->Cast<LogicalCTERef>();
	auto &unnest_side = join.children[1 - domain_side];
	if (unnest_side->type != LogicalOperatorType::LOGICAL_GET || unnest_side->children.size() != 1) {
		return false;
	}
	auto &get = unnest_side->Cast<LogicalGet>();
	if (!ExpressionBinder::IsUnnestFunction(get.function.GetName()) || get.ordinality_idx.IsValid() ||
	    get.children[0]->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return false;
	}
	auto &arg_proj = get.children[0]->Cast<LogicalProjection>();
	if (arg_proj.children.size() != 1 || !IsCTERef(*arg_proj.children[0], dedup_cte.table_index)) {
		return false;
	}
	// The argument runs once per dedup group here and would run once per domain row after the rewrite: a draw in it
	// (string_split(x || random()::VARCHAR, ' ')) would give each duplicate row its own value
	if (AIOperatorHasVolatileNonAI(arg_proj)) {
		return false;
	}
	const auto dedup_scan_index = arg_proj.children[0]->Cast<LogicalCTERef>().table_index;

	// Each condition pairs a domain column with the dedup column grouping that same domain column, so every
	// domain row matches exactly one dedup row. Map each dedup column to its domain scan column.
	std::unordered_map<idx_t, ColumnBinding> dedup_to_domain; // dedup scan column -> domain scan binding
	for (auto &cond : join.conditions) {
		if (!cond.IsComparison() || cond.GetComparisonType() != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			return false;
		}
		auto lhs = AsColumnRef(cond.GetLHS());
		auto rhs = AsColumnRef(cond.GetRHS());
		if (!lhs || !rhs) {
			return false;
		}
		auto domain_col = lhs->Binding().table_index == domain_scan.table_index ? lhs : rhs;
		auto other_col = domain_col.get() == lhs.get() ? rhs : lhs;
		if (domain_col->Binding().table_index != domain_scan.table_index ||
		    other_col->Binding().table_index != arg_proj.table_index) {
			return false;
		}
		const idx_t k = other_col->Binding().column_index.GetIndex();
		if (k >= arg_proj.expressions.size()) {
			return false;
		}
		auto passed = AsColumnRef(*arg_proj.expressions[k]);
		if (!passed || passed->Binding().table_index != dedup_scan_index) {
			return false;
		}
		const idx_t dedup_col = passed->Binding().column_index.GetIndex();
		if (dedup_col >= dedup.groups.size()) {
			return false;
		}
		auto grouped = DomainColumnOfGroup(*dedup.groups[dedup_col], domain_ref, *body_it->second);
		if (!grouped.IsValid() || grouped.GetIndex() != domain_col->Binding().column_index.GetIndex()) {
			return false;
		}
		dedup_to_domain[dedup_col] = domain_col->Binding();
	}
	if (dedup_to_domain.size() != dedup.groups.size()) {
		return false;
	}
	auto to_domain = [&](unique_ptr<Expression> &expr) -> bool {
		bool ok = true;
		std::function<void(unique_ptr<Expression> &)> rebind = [&](unique_ptr<Expression> &e) {
			if (e->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
				auto &colref = e->Cast<BoundColumnRefExpression>();
				auto it = colref.Binding().table_index == dedup_scan_index
				              ? dedup_to_domain.find(colref.Binding().column_index.GetIndex())
				              : dedup_to_domain.end();
				if (colref.Depth() != 0 || it == dedup_to_domain.end()) {
					ok = false;
					return;
				}
				colref.BindingMutable() = it->second;
				return;
			}
			ExpressionIterator::EnumerateChildren(*e, [&](unique_ptr<Expression> &child) { rebind(child); });
		};
		rebind(expr);
		return ok;
	};

	// The GET's own columns are the unnested values of its input; its other columns pass the input through.
	unnest_side->ResolveOperatorTypes();
	const auto get_bindings = unnest_side->GetColumnBindings();
	const auto get_index = unnest_side->GetTableIndex()[0];
	auto unnest = make_uniq<LogicalUnnest>(binder.GenerateTableIndex());
	ColumnBindingReplacer replacer;
	for (idx_t i = 0; i < get_bindings.size(); i++) {
		const auto &binding = get_bindings[i];
		const auto &type = unnest_side->types[i];
		const idx_t k = binding.column_index.GetIndex();
		if (k >= arg_proj.expressions.size()) {
			return false;
		}
		auto expr = arg_proj.expressions[k]->Copy();
		if (!to_domain(expr)) {
			return false;
		}
		if (binding.table_index == get_index) {
			auto unnest_expr = make_uniq<BoundUnnestExpression>(type);
			unnest_expr->ChildMutable() = std::move(expr);
			const auto pos = ColumnBinding::PushExpression(unnest->expressions, std::move(unnest_expr));
			replacer.replacement_bindings.emplace_back(binding, ColumnBinding(unnest->unnest_index, pos), type);
		} else if (binding.table_index == arg_proj.table_index) {
			auto passed = AsColumnRef(*expr);
			if (!passed) {
				return false; // a computed pass-through column has no domain column to stand in for it
			}
			replacer.replacement_bindings.emplace_back(binding, passed->Binding(), type);
		} else {
			return false;
		}
	}
	if (unnest->expressions.empty()) {
		return false;
	}

	unnest->children.push_back(std::move(join.children[domain_side]));
	*join_slot = std::move(unnest);
	dedup_slot = std::move(dedup_cte.children[1]);
	replacer.VisitOperator(*root);
	return true;
}

static bool RewriteTree(Binder &binder, unique_ptr<LogicalOperator> &root, unique_ptr<LogicalOperator> &op,
                        const std::unordered_map<idx_t, LogicalOperator *> &bodies) {
	bool changed = false;
	for (auto &child : op->children) {
		changed = RewriteTree(binder, root, child, bodies) || changed;
	}
	return RewriteOne(binder, root, op, bodies) || changed;
}

bool AIRewriteLateralUnnests(Binder &binder, unique_ptr<LogicalOperator> &plan) {
	std::unordered_map<idx_t, LogicalOperator *> bodies;
	CollectCTEBodies(*plan, bodies);
	const bool changed = RewriteTree(binder, plan, plan, bodies);
	if (changed) {
		plan->ResolveOperatorTypes();
	}
	return changed;
}

} // namespace duckdb
