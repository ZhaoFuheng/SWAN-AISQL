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

bool SemanticFilterPullup::IsClusterNode(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return op.Cast<LogicalComparisonJoin>().join_type == JoinType::INNER;
	}
	if (op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		return op.Cast<LogicalAnyJoin>().join_type == JoinType::INNER;
	}
	return op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT;
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

void SemanticFilterPullup::ExtractAIFilters(unique_ptr<LogicalOperator> &op,
                                            vector<unique_ptr<Expression>> &pulled, idx_t cluster_card) {
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
		// One combined speculative pre-filter over the AND of the pulled leaves (if any) -- unless the join
		// above this leaf is SELECTIVE (reduces its rows). Once pulled up they run on the join-reduced set,
		// so pre-evaluating at the leaf would spend LLM calls on rows the join discards. DuckDB systematically
		// UNDER-estimates join cardinality, so we bypass only on a strong reduction signal: estimated fanout
		// (join output / leaf input) below AI_SPECULATIVE_MIN_FANOUT (default 0.3). When bypassed we leave
		// NOTHING at the leaf (== pure pull-up); the pulled-up predicate is still result-complete.
		unique_ptr<Expression> spec;
		if (!leaves.empty()) {
			const idx_t leaf_card = op->children.empty()
			                            ? op->EstimateCardinality(optimizer.context)
			                            : op->children[0]->EstimateCardinality(optimizer.context);
			const double fanout =
			    leaf_card ? static_cast<double>(cluster_card) / static_cast<double>(leaf_card) : 1.0;
			const double min_fanout =
			    AIDoubleSetting(optimizer.context, "ai_debug_speculative_min_fanout", 0.3);
			// At threshold 0 the speculative gate passes every row through (pure pull-up) but the node still
			// embeds/predicts each row -- pure overhead. Read the same AI_SPECULATIVE_THRESHOLD the node binds
			// (default 0.5) and skip emitting it entirely when it is <= 0; the pulled-up filter is complete.
			const double spec_threshold =
			    AIDoubleSetting(optimizer.context, "ai_debug_speculative_threshold", 0.5);
			// The fanout gate exists because a speculative leaf pre-filter would spend LLM calls on rows a
			// selective join discards. But Yannakakis now reduces the base RELATIONALLY before this node, so the
			// leaf already sees only join-surviving rows -- making the gate obsolete. AI_SPECULATIVE_ALWAYS=1
			// drops the fanout gate to measure that (the threshold<=0 "pure pull-up" bypass still applies).
			const bool always_spec =
			    AIBoolSetting(optimizer.context, "ai_debug_speculative_always", false);
			const bool bypass = (!always_spec && fanout < min_fanout) || spec_threshold <= 0.0;
			if (AIConfig::Get().debug_log.find("spec") != string::npos) {
				fprintf(stderr,
				        "[spec-debug] leaf_card=%llu cluster_card=%llu fanout=%.3f min=%.2f thr=%.2f -> %s\n",
				        (unsigned long long)leaf_card, (unsigned long long)cluster_card, fanout, min_fanout,
				        spec_threshold, bypass ? "BYPASS (pure pull-up)" : "emit speculative");
			}
			if (!bypass) {
				spec = BuildSpeculativeCall(leaves); // reads leaves' pointers -> must precede the move below
			}
		}
		// Pull the AI predicates up (they run above the join); everything else stays, plus the pre-filter.
		vector<unique_ptr<Expression>> keep;
		keep.reserve(filter.expressions.size() + 1);
		for (idx_t i = 0; i < filter.expressions.size(); i++) {
			if (pull[i]) {
				pulled.push_back(std::move(filter.expressions[i]));
			} else {
				keep.push_back(std::move(filter.expressions[i]));
			}
		}
		if (spec) {
			keep.push_back(std::move(spec));
		}
		filter.expressions = std::move(keep);
	} else if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		// A multi-table AI predicate (ai_filter / ai_classify=/ai_score>/ai_complete=) is folded into the
		// join as a (non-comparison) condition -- its expression references both tables, so it can only run
		// AFTER the join. Lift every such condition into a filter above the join (where reorder + streaming
		// dedup then handle it); relational equi-conditions stay in the join. No speculative placeholder --
		// a two-table predicate has no lower position to pre-reduce. If ALL conditions were AI (a join on AI
		// conditions only, no relational key), the inner join degenerates to a cross product.
		auto &join = op->Cast<LogicalComparisonJoin>();
		idx_t ai_conds = 0;
		for (auto &cond : join.conditions) {
			if (!cond.IsComparison() && ExprContainsAICall(cond.GetJoinExpression())) {
				ai_conds++;
			}
		}
		if (ai_conds > 0) {
			vector<JoinCondition> keep_conds;
			for (auto &cond : join.conditions) {
				if (!cond.IsComparison() && ExprContainsAICall(cond.GetJoinExpression())) {
					pulled.push_back(std::move(cond.JoinExpressionReference()));
				} else {
					keep_conds.push_back(std::move(cond));
				}
			}
			if (keep_conds.empty()) {
				// No relational condition remains -> a bare cross product (the lifted filter re-applies the
				// AI predicate above it, so the result is unchanged).
				op = LogicalCrossProduct::Create(std::move(op->children[0]), std::move(op->children[1]));
			} else {
				join.conditions = std::move(keep_conds);
			}
		}
	} else if (op->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		// A no-equi INNER join whose arbitrary condition involves AI predicates (the "join on AI conditions
		// only" case): lift the whole condition above the join and degenerate to a cross product -- inner
		// join on C == cross product then filter C. For a pure-AI condition (the common case) the lifted
		// filter then reorders/dedups; a mixed AI+relational condition is still correct, just evaluated above.
		auto &join = op->Cast<LogicalAnyJoin>();
		if (join.condition && ExprContainsAICall(*join.condition)) {
			pulled.push_back(std::move(join.condition));
			op = LogicalCrossProduct::Create(std::move(op->children[0]), std::move(op->children[1]));
		}
	}

	// Descend through operators between the filter and the cluster top. A projection RENAMES and
	// may DROP the pulled filter's source columns, and a join's projection maps may drop them too,
	// so after recursing, every expression pulled from below is ROUTED through this operator:
	// colrefs rebind to projection outputs (appending passthrough columns the projection dropped)
	// and join maps widen to keep referenced child columns.
	const idx_t pulled_before = pulled.size();
	switch (op->type) {
	case LogicalOperatorType::LOGICAL_FILTER:
	case LogicalOperatorType::LOGICAL_PROJECTION:
	case LogicalOperatorType::LOGICAL_GET:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		for (auto &child : op->children) {
			ExtractAIFilters(child, pulled, cluster_card);
		}
		break;
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		// A materialized CTE node sits BETWEEN the cluster top and the join tree, and stopping here
		// hides every filter below it (agent_bench Q26: the walk found 1 of 4 AI filters, so part
		// and supplier were evaluated on their full base tables instead of on join survivors).
		// Descend into the MAIN QUERY (children[1]) only. The CTE BODY (children[0]) is off limits:
		// lifting a predicate out of a body that other references also read would drop the filter
		// for those references.
		if (op->children.size() > 1) {
			ExtractAIFilters(op->children[1], pulled, cluster_card);
		}
		break;
	default:
		break; // stop at aggregates / set ops / subqueries -- do not pull a filter across them
	}
	if (pulled.size() > pulled_before) {
		if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &proj = op->Cast<LogicalProjection>();
			bool appended = false;
			for (idx_t i = pulled_before; i < pulled.size(); i++) {
				RouteThroughProjection(pulled[i], proj, appended);
			}
			if (appended) {
				proj.ResolveOperatorTypes();
			}
		} else if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		           op->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
			auto &join = op->Cast<LogicalJoin>();
			for (idx_t i = pulled_before; i < pulled.size(); i++) {
				WidenJoinMaps(*pulled[i], join);
			}
		}
	}
}

void SemanticFilterPullup::PullupTree(unique_ptr<LogicalOperator> &op_slot) {
	if (IsClusterNode(*op_slot)) {
		// Outermost cluster top: pull every ai_filter within the cluster subtree above it.
		const idx_t cluster_card = op_slot->EstimateCardinality(optimizer.context);
		vector<unique_ptr<Expression>> pulled;
		ExtractAIFilters(op_slot, pulled, cluster_card);
		if (!pulled.empty()) {
			auto pull_filter = make_uniq<LogicalFilter>();
			for (auto &expr : pulled) {
				pull_filter->expressions.push_back(std::move(expr));
			}
			pull_filter->children.push_back(std::move(op_slot));
			pull_filter->ResolveOperatorTypes();
			op_slot = std::move(pull_filter);
		}
		return; // v1: handle the outermost cluster only
	}
	for (auto &child : op_slot->children) {
		PullupTree(child);
	}
}

unique_ptr<LogicalOperator> SemanticFilterPullup::Optimize(unique_ptr<LogicalOperator> op) {
	if (!AIBoolSetting(optimizer.context, "ai_pullup", true)) {
		return op;
	}
	PullupTree(op);
	return op;
}

} // namespace duckdb
