#include "optimizer/semantic_filter_pullup.hpp"

#include "ai_client.hpp"
#include "ai_settings.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/common/error_data.hpp"
#include "filter_tree_order.hpp"
#include "duckdb/function/function_binder.hpp"
#include "optimizer/ai_filter_tree_build.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"

#include <cstdio>
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"

#include <cstdlib>

namespace duckdb {

static bool IsAICallExpr(const Expression &expr) {
	if (expr.GetExpressionType() != ExpressionType::BOUND_FUNCTION) {
		return false;
	}
	const auto &name = expr.Cast<BoundFunctionExpression>().Function().GetName();
	return name == "ai_filter" || name == "ai_classify" || name == "ai_score" || name == "ai_complete";
}

//! Does `expr` contain any AI-function call (ai_filter / ai_classify / ai_score / ai_complete)? Used to
//! decide whether a multi-table join condition should be lifted above the join.
static bool ExprContainsAICall(const Expression &expr) {
	if (IsAICallExpr(expr)) {
		return true;
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!found && ExprContainsAICall(child)) {
			found = true;
		}
	});
	return found;
}

unique_ptr<Expression> SemanticFilterPullup::BuildSpeculativeCall(vector<AIMixedLeaf> &leaves) {
	const idx_t m = leaves.size();
	// Combined boolean tree = AND of the pulled leaves (a single leaf when m == 1). The pre-filter
	// prunes rows likely to fail the whole conjunction; each pulled predicate is a valid single-table
	// pushdown here, so dropping a confirmed-false row is sound.
	unique_ptr<AIFilterTreeNode> tree;
	if (m == 1) {
		tree = AIFilterTreeNode::Leaf(0);
	} else {
		vector<unique_ptr<AIFilterTreeNode>> children;
		children.reserve(m);
		for (idx_t j = 0; j < m; j++) {
			children.push_back(AIFilterTreeNode::Leaf(j));
		}
		tree = AIFilterTreeNode::Op(AIFilterTreeType::AND_OP, std::move(children));
	}
	const string tree_str = AIFilterTreeSerialize(*tree);

	// Per leaf: bake the scalar-exact call prompt, the (predicate, input) MLP-feature split, and the meta
	// token (empty/all-F for a pure ai_filter tree). Shared with the reorder rewrite.
	auto &context = optimizer.context;
	vector<unique_ptr<Expression>> call_prompts, preds, inputs;
	string meta_str;
	if (!AIBuildMixedLeafArgs(context, leaves, call_prompts, preds, inputs, meta_str)) {
		return nullptr; // a leaf was not bakeable -> emit no pre-filter (the pulled-up predicate is complete)
	}

	// speculative_ai_function_with_embed(tree_str, call_0..{m-1}, pred_0..{m-1}, input_0..{m-1}, meta_str).
	vector<unique_ptr<Expression>> args;
	args.reserve(2 + 3 * m);
	args.push_back(make_uniq<BoundConstantExpression>(Value(tree_str)));
	for (idx_t j = 0; j < m; j++) {
		args.push_back(std::move(call_prompts[j]));
	}
	for (idx_t j = 0; j < m; j++) {
		args.push_back(std::move(preds[j]));
	}
	for (idx_t j = 0; j < m; j++) {
		args.push_back(std::move(inputs[j]));
	}
	args.push_back(make_uniq<BoundConstantExpression>(Value(meta_str)));
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto &entry = catalog.GetEntry<ScalarFunctionCatalogEntry>(
	    context, QualifiedName(catalog.GetName(), Identifier::DefaultSchema(), "speculative_ai_function_with_embed"));
	FunctionBinder function_binder(context);
	ErrorData error;
	return function_binder.BindScalarFunction(entry, std::move(args), error);
}

//! Rebind a pulled expression's colrefs through `proj`: a colref matching a passthrough output
//! rebinds to it; a colref the projection dropped is APPENDED as a new passthrough output (safe:
//! existing output indices are untouched, consumers above are unaffected).
static void RouteThroughProjection(unique_ptr<Expression> &expr, LogicalProjection &proj, bool &appended) {
	if (expr->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto &colref = expr->Cast<BoundColumnRefExpression>();
		if (colref.Binding().table_index == proj.table_index) {
			return; // already routed (multiple pulled exprs may share colrefs)
		}
		for (idx_t i = 0; i < proj.expressions.size(); i++) {
			auto &pe = proj.expressions[i];
			if (pe->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
			    pe->Cast<BoundColumnRefExpression>().Binding() == colref.Binding()) {
				expr = make_uniq<BoundColumnRefExpression>(colref.GetAlias(), pe->GetReturnType(),
				                                           ColumnBinding(proj.table_index, ProjectionIndex(i)));
				return;
			}
		}
		const auto type = colref.GetReturnType();
		const auto below = colref.Binding();
		proj.expressions.push_back(make_uniq<BoundColumnRefExpression>(type, below));
		appended = true;
		expr = make_uniq<BoundColumnRefExpression>(
		    colref.GetAlias(), type, ColumnBinding(proj.table_index, ProjectionIndex(proj.expressions.size() - 1)));
		return;
	}
	ExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<Expression> &child) { RouteThroughProjection(child, proj, appended); });
}

//! Ensure every child-side column a pulled expression references survives `join`'s projection
//! maps. Join output bindings ARE child bindings, so appending to a map never renumbers existing
//! consumers; an empty map already passes everything through.
static void WidenJoinMaps(Expression &expr, LogicalJoin &join) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto &colref = expr.Cast<BoundColumnRefExpression>();
		auto ensure = [&](vector<ProjectionIndex> &map, const vector<ColumnBinding> &bindings) {
			if (map.empty()) {
				return; // empty = full passthrough
			}
			for (idx_t i = 0; i < bindings.size(); i++) {
				if (bindings[i] == colref.Binding()) {
					for (auto &m : map) {
						if (m.GetIndex() == i) {
							return; // already kept
						}
					}
					map.push_back(ProjectionIndex(i));
					return;
				}
			}
		};
		if (!join.children.empty()) {
			ensure(join.left_projection_map, join.children[0]->GetColumnBindings());
		}
		if (join.children.size() > 1) {
			ensure(join.right_projection_map, join.children[1]->GetColumnBindings());
		}
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { WidenJoinMaps(child, join); });
}

//! Does every column `expr` reads already pass through `proj` (so routing needs no appended output)?
static bool ProjectionPassesAll(const Expression &expr, const LogicalProjection &proj) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		const auto &binding = expr.Cast<BoundColumnRefExpression>().Binding();
		if (binding.table_index == proj.table_index) {
			return true;
		}
		for (auto &pe : proj.expressions) {
			if (pe->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
			    pe->Cast<BoundColumnRefExpression>().Binding() == binding) {
				return true;
			}
		}
		return false;
	}
	bool all = true;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (all && !ProjectionPassesAll(child, proj)) {
			all = false;
		}
	});
	return all;
}

//! Is `op` a join whose conditions may carry AI predicates we lift (inner joins only: lifting a
//! condition out of an outer, semi or mark join would change which rows survive it)?
static bool IsInnerJoin(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN || op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		return op.Cast<LogicalJoin>().join_type == JoinType::INNER;
	}
	return op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT;
}

//! The commutation law: may a filter on the columns of `op`'s child `child_idx` move above `op`?
//! `reduces` reports whether `op` removes rows of that child (the lift is then worth keeping).
static bool Commutes(const LogicalOperator &op, idx_t child_idx, bool &reduces) {
	reduces = false;
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_FILTER:
		// a relational conjunct above the predicate's site removes rows before it runs
		reduces = !op.expressions.empty();
		return true;
	case LogicalOperatorType::LOGICAL_PROJECTION:
		return true; // the caller checks that the columns can be routed through
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		reduces = true;
		return true;
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN: {
		// a filter on one side commutes with the join iff that side is preserved; it is worth lifting
		// iff the join can drop that side's rows (inner, semi and anti joins; not outer, single, mark)
		const auto join_type = op.Cast<LogicalJoin>().join_type;
		switch (join_type) {
		case JoinType::INNER:
			reduces = true;
			return true;
		case JoinType::SEMI:
		case JoinType::ANTI:
			reduces = child_idx == 0;
			return child_idx == 0;
		case JoinType::RIGHT_SEMI:
		case JoinType::RIGHT_ANTI:
			reduces = child_idx == 1;
			return child_idx == 1;
		case JoinType::LEFT:
		case JoinType::SINGLE:
		case JoinType::MARK:
			return child_idx == 0;
		case JoinType::RIGHT:
			return child_idx == 1;
		default:
			return false; // OUTER and anything unknown: neither side is safe
		}
	}
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		// the main query (child 1) only; a predicate lifted out of the body would be lost to the other
		// references of the CTE
		return child_idx == 1;
	default:
		return false; // aggregates, limits, sorts, set operations, windows, dependent joins: stop here
	}
}

//! May `op`'s child `child_idx` grow extra output columns (a routed predicate's passthrough column)
//! without changing what `op` produces? Positional consumers (set operations, DISTINCT over all
//! columns, a CTE body read by position) cannot take them; an operator that re-emits its own bindings
//! (projection, aggregate) hides them; everything else passes them on to its own parent.
static bool ChildMayGrow(const LogicalOperator &op, idx_t child_idx, bool op_may_grow) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_PROJECTION:
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		return true;
	case LogicalOperatorType::LOGICAL_FILTER:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
	case LogicalOperatorType::LOGICAL_ORDER_BY:
	case LogicalOperatorType::LOGICAL_LIMIT:
	case LogicalOperatorType::LOGICAL_TOP_N:
		return op_may_grow;
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		return child_idx == 1 && op_may_grow;
	default:
		return false;
	}
}

void SemanticFilterPullup::Extract(unique_ptr<LogicalOperator> &op, vector<PulledGroup> &groups) {
	if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = op->Cast<LogicalFilter>();
		// Detect the AI-comparison leaves (ai_filter / ai_classify=/ai_score>/ai_complete=) among the ANDed
		// predicates. The leaves hold pointers INTO filter.expressions, so build the pre-filter (which reads
		// them) BEFORE moving the pulled exprs up.
		vector<AIMixedLeaf> leaves;
		vector<char> pull(filter.expressions.size(), 0);
		for (idx_t i = 0; i < filter.expressions.size(); i++) {
			AIMixedLeaf leaf;
			if (AIDetectMixedLeaf(*filter.expressions[i], leaf)) {
				leaves.push_back(leaf);
				pull[i] = 1;
			}
		}
		if (leaves.empty()) {
			return;
		}
		PulledGroup group;
		group.origin = &filter;
		// One combined speculative pre-filter over the AND of the pulled leaves. It is installed at the
		// origin only if the lift crosses an operator that removes rows (Settle), so it spends LLM calls
		// only on rows the join would keep: the semi-join reduction has already pruned the leaf
		// relationally. ai_speculative=false leaves nothing at the leaf (pure pull-up); the pulled-up
		// predicate is complete either way.
		if (AIBoolSetting(optimizer.context, "ai_speculative", true)) {
			group.speculative = BuildSpeculativeCall(leaves); // reads leaves' pointers -> precedes the move
		}
		vector<unique_ptr<Expression>> keep;
		keep.reserve(filter.expressions.size());
		for (idx_t i = 0; i < filter.expressions.size(); i++) {
			if (pull[i]) {
				group.originals.push_back(filter.expressions[i]->Copy());
				group.origin_positions.push_back(i);
				group.exprs.push_back(std::move(filter.expressions[i]));
			} else {
				keep.push_back(std::move(filter.expressions[i]));
			}
		}
		filter.expressions = std::move(keep);
		groups.push_back(std::move(group));
	} else if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN && IsInnerJoin(*op)) {
		// A multi-table AI predicate (ai_filter / ai_classify=/ai_score>/ai_complete=) is folded into the
		// join as a (non-comparison) condition -- its expression references both tables, so it can only run
		// AFTER the join. Lift every such condition into a filter above the join (where reorder + streaming
		// dedup then handle it); relational equi-conditions stay in the join. No speculative placeholder --
		// a two-table predicate has no lower position to pre-reduce. If ALL conditions were AI (a join on AI
		// conditions only, no relational key), the inner join degenerates to a cross product.
		auto &join = op->Cast<LogicalComparisonJoin>();
		PulledGroup group;
		vector<JoinCondition> keep_conds;
		for (auto &cond : join.conditions) {
			if (!cond.IsComparison() && ExprContainsAICall(cond.GetJoinExpression())) {
				group.exprs.push_back(std::move(cond.JoinExpressionReference()));
			} else {
				keep_conds.push_back(std::move(cond));
			}
		}
		if (group.exprs.empty()) {
			join.conditions = std::move(keep_conds); // nothing lifted: the conditions go back untouched
			return;
		}
		if (keep_conds.empty()) {
			// No relational condition remains -> a bare cross product (the lifted filter re-applies the
			// AI predicate above it, so the result is unchanged).
			op = LogicalCrossProduct::Create(std::move(op->children[0]), std::move(op->children[1]));
		} else {
			join.conditions = std::move(keep_conds);
		}
		// the join itself is the row-removing operator below the lifted site
		group.checkpoint = op.get();
		for (auto &expr : group.exprs) {
			group.checkpoint_exprs.push_back(expr->Copy());
		}
		groups.push_back(std::move(group));
	} else if (op->type == LogicalOperatorType::LOGICAL_ANY_JOIN && IsInnerJoin(*op)) {
		// A no-equi INNER join whose arbitrary condition involves AI predicates (the "join on AI conditions
		// only" case): lift the whole condition above the join and degenerate to a cross product -- inner
		// join on C == cross product then filter C. For a pure-AI condition (the common case) the lifted
		// filter then reorders/dedups; a mixed AI+relational condition is still correct, just evaluated above.
		auto &join = op->Cast<LogicalAnyJoin>();
		if (join.condition && ExprContainsAICall(*join.condition)) {
			PulledGroup group;
			group.exprs.push_back(std::move(join.condition));
			op = LogicalCrossProduct::Create(std::move(op->children[0]), std::move(op->children[1]));
			group.checkpoint = op.get();
			group.checkpoint_exprs.push_back(group.exprs[0]->Copy());
			groups.push_back(std::move(group));
		}
	}
}

void SemanticFilterPullup::InstallAbove(unique_ptr<LogicalOperator> &slot, vector<unique_ptr<Expression>> &exprs) {
	if (slot->type != LogicalOperatorType::LOGICAL_FILTER || !settled.count(&slot->Cast<LogicalFilter>())) {
		auto filter = make_uniq<LogicalFilter>();
		filter->children.push_back(std::move(slot));
		slot = std::move(filter);
		settled.insert(&slot->Cast<LogicalFilter>());
	}
	auto &site = slot->Cast<LogicalFilter>();
	for (auto &expr : exprs) {
		site.expressions.push_back(std::move(expr));
	}
	site.ResolveOperatorTypes();
}

//! The child slot of `from`'s subtree that holds `target` (and the operator owning that slot), or none.
static unique_ptr<LogicalOperator> *FindSlot(LogicalOperator &from, const LogicalOperator &target,
                                             optional_ptr<LogicalOperator> &parent) {
	for (auto &child : from.children) {
		if (child.get() == &target) {
			parent = &from;
			return &child;
		}
		if (auto *slot = FindSlot(*child, target, parent)) {
			return slot;
		}
	}
	return nullptr;
}

void SemanticFilterPullup::Settle(LogicalOperator &from, PulledGroup &group) {
	if (!group.checkpoint) {
		// Nothing crossed removes rows: the lift would not save a call. Put the predicates back exactly
		// as written (the fold stage still gives them a filter of their own).
		D_ASSERT(group.origin);
		for (idx_t i = 0; i < group.originals.size(); i++) {
			auto &exprs = group.origin->expressions;
			exprs.insert(exprs.begin() + NumericCast<int64_t>(group.origin_positions[i]),
			             std::move(group.originals[i]));
		}
		return;
	}
	optional_ptr<LogicalOperator> parent;
	auto *slot = FindSlot(from, *group.checkpoint, parent);
	if (!slot) {
		throw InternalException("SemanticFilterPullup: a group's checkpoint is not below the operator it settles at");
	}
	if (parent->type == LogicalOperatorType::LOGICAL_FILTER && settled.count(&parent->Cast<LogicalFilter>())) {
		// an earlier group already settled above this checkpoint: share its filter
		auto &site = parent->Cast<LogicalFilter>();
		for (auto &expr : group.checkpoint_exprs) {
			site.expressions.push_back(std::move(expr));
		}
		site.ResolveOperatorTypes();
	} else {
		InstallAbove(*slot, group.checkpoint_exprs);
	}
	if (group.origin && group.speculative) {
		group.origin->expressions.push_back(std::move(group.speculative));
	}
}

vector<SemanticFilterPullup::PulledGroup> SemanticFilterPullup::Lift(unique_ptr<LogicalOperator> &op,
                                                                     bool extra_columns_ok, bool is_root) {
	// Bottom-up: each child hands back the groups that commute through it.
	vector<vector<PulledGroup>> from_child(op->children.size());
	for (idx_t i = 0; i < op->children.size(); i++) {
		from_child[i] = Lift(op->children[i], ChildMayGrow(*op, i, extra_columns_ok), false);
	}
	// The predicates `op` itself holds start their lift from here (a join's AI conditions may turn
	// `op` into a cross product, so this runs before the commutation checks below read `op`).
	vector<PulledGroup> outgoing;
	Extract(op, outgoing);

	for (idx_t i = 0; i < from_child.size(); i++) {
		for (auto &group : from_child[i]) {
			bool reduces = false;
			bool commutes = !is_root && Commutes(*op, i, reduces);
			if (commutes && op->type == LogicalOperatorType::LOGICAL_PROJECTION && !extra_columns_ok) {
				// routing may need an appended passthrough column, which this projection may not grow
				auto &proj = op->Cast<LogicalProjection>();
				for (auto &expr : group.exprs) {
					if (!ProjectionPassesAll(*expr, proj)) {
						commutes = false;
						break;
					}
				}
			}
			if (!commutes) {
				Settle(*op, group);
				continue;
			}
			// Route the group to op's output bindings: a projection renames and may drop its columns, a
			// join's projection maps may drop them; filters and CTE nodes pass bindings through unchanged.
			if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &proj = op->Cast<LogicalProjection>();
				bool appended = false;
				for (auto &expr : group.exprs) {
					RouteThroughProjection(expr, proj, appended);
				}
				if (appended) {
					proj.ResolveOperatorTypes();
				}
			} else if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
			           op->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
				auto &join = op->Cast<LogicalJoin>();
				for (auto &expr : group.exprs) {
					WidenJoinMaps(*expr, join);
				}
			}
			if (reduces) {
				group.checkpoint = op.get();
				group.checkpoint_exprs.clear();
				for (auto &expr : group.exprs) {
					group.checkpoint_exprs.push_back(expr->Copy());
				}
			}
			outgoing.push_back(std::move(group));
		}
	}
	return outgoing;
}

unique_ptr<LogicalOperator> SemanticFilterPullup::Optimize(unique_ptr<LogicalOperator> op) {
	if (!AIBoolSetting(optimizer.context, "ai_pullup", true)) {
		return op;
	}
	// The root never commutes anything: every group settles at its checkpoint or returns to its origin.
	auto groups = Lift(op, false, true);
	for (auto &group : groups) {
		if (group.checkpoint.get() == op.get()) {
			InstallAbove(op, group.checkpoint_exprs); // the root itself was a join holding AI conditions
		} else {
			Settle(*op, group);
		}
	}
	return op;
}

} // namespace duckdb
