#include "optimizer/ai_positional_maps.hpp"
#include "optimizer/ai_join_rewrite.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/function/function_binder.hpp"
#include "filter_tree_order.hpp"
#include "optimizer/ai_filter_tree_build.hpp"
#include "plan/logical_ai_factor_graph.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_order.hpp"

#include "ai_settings.hpp"

#include <functional>
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/common/projection_index.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "plan/logical_ai_region.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace duckdb {

namespace {

// Rewrites every reference to `from` into `to`, in operator expressions AND join conditions. Run over the whole
// plan BEFORE the projection's passthrough to `from` is appended, so at replace time the only references to
// `from` are the true consumers above the pushed region (nothing below the projection references it).
class AIRegionBindingReplacer : public LogicalOperatorVisitor {
public:
	AIRegionBindingReplacer(ColumnBinding from, ColumnBinding to) : from(from), to(to) {
	}
	void VisitOperator(LogicalOperator &op) override {
		VisitOperatorChildren(op);
		VisitOperatorExpressions(op);
	}

protected:
	unique_ptr<Expression> VisitReplace(BoundColumnRefExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		if (expr.Binding() == from) {
			expr.BindingMutable() = to;
		}
		return nullptr;
	}

private:
	ColumnBinding from;
	ColumnBinding to;
};

} // namespace

// Collect every column binding the expression tree reads.
static void CollectColumnRefs(const Expression &expr, vector<ColumnBinding> &out) {
	if (expr.GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
		out.push_back(expr.Cast<BoundColumnRefExpression>().Binding());
	}
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) { CollectColumnRefs(child, out); });
}

// True iff every binding in `keys` is produced by `side`.
static bool AllContainedIn(const vector<ColumnBinding> &keys, const vector<ColumnBinding> &side) {
	for (auto &k : keys) {
		if (std::find(side.begin(), side.end(), k) == side.end()) {
			return false;
		}
	}
	return true;
}

// Replace every reference to one of `proj`'s output columns with a copy of the projected expression itself, so
// the expression can be evaluated directly against the projection's child. Fails (returns false) if any ref
// points elsewhere -- the caller then leaves the plan untouched.
static bool InlineProjectionRefs(unique_ptr<Expression> &expr, const LogicalProjection &proj) {
	if (expr->GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
		auto &binding = expr->Cast<BoundColumnRefExpression>().Binding();
		if (binding.table_index.index != proj.table_index.index) {
			return false;
		}
		const idx_t col = binding.column_index.GetIndex();
		if (col >= proj.expressions.size()) {
			return false;
		}
		expr = proj.expressions[col]->Copy();
		return true;
	}
	bool ok = true;
	ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
		if (ok && !InlineProjectionRefs(child, proj)) {
			ok = false;
		}
	});
	return ok;
}

// The Yannakakis semi-join reducer marks its relational reduction with a SEMI join. Pushing the region below an
// INNER join is call-optimal only when the target side is already reduced to the keys that survive the join --
// otherwise the region would evaluate distinct inputs the join was about to discard (Stage 1 above the join
// never pays for those). The reduction itself is purely relational: it never contains or invokes an AI call.
static bool SubtreeHasSemiReduction(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		const auto jt = op.Cast<LogicalComparisonJoin>().join_type;
		if (jt == JoinType::SEMI || jt == JoinType::RIGHT_SEMI) {
			return true;
		}
	}
	for (auto &child : op.children) {
		if (SubtreeHasSemiReduction(*child)) {
			return true;
		}
	}
	return false;
}

// The operator whose child is `target`, or null.
static optional_ptr<LogicalOperator> ParentOf(LogicalOperator &root, LogicalOperator &target) {
	for (auto &child : root.children) {
		if (child.get() == &target) {
			return &root;
		}
		if (auto found = ParentOf(*child, target)) {
			return found;
		}
	}
	return nullptr;
}

//! A FILTER/ORDER_BY above `target` may carry a POSITIONAL projection_map built for `target`'s old output
//! order. The push-below reshapes that order (the region result moves from the end into the pushed side);
//! binding identities are preserved, only their sequence changes, so the map is remapped by identity.
//! The map need not sit on `target`'s direct parent: a map-less filter/order passes its child's order through,
//! and an AI region emits its child's columns in order and appends its own result, so the reshape is carried
//! up through those until a map is reached. A region stacked above the pushed one is exactly the
//! ai_reorder=false shape -- two unfolded filters over one join, the filter testing both results on top.
static void RemapParentPositionalMaps(LogicalOperator &root, LogicalOperator &target, vector<ColumnBinding> old_out) {
	optional_ptr<LogicalOperator> cur = &target;
	while (auto parent = ParentOf(root, *cur)) {
		if (parent->type == LogicalOperatorType::LOGICAL_FILTER ||
		    parent->type == LogicalOperatorType::LOGICAL_ORDER_BY) {
			auto &map = parent->type == LogicalOperatorType::LOGICAL_FILTER
			                ? parent->Cast<LogicalFilter>().projection_map
			                : parent->Cast<LogicalOrder>().projection_map;
			if (!map.empty()) {
				PositionalMapSnapshot::RemapByIdentity(map, old_out, cur->GetColumnBindings());
				return; // above here the order is defined by this map, now correct
			}
		} else if (LogicalAIRegion::TryCast(*parent)) {
			const auto out = parent->GetColumnBindings(); // child order, then the region's own result
			for (idx_t i = old_out.size(); i < out.size(); i++) {
				old_out.push_back(out[i]);
			}
		} else {
			return; // a projection (or anything else) re-expresses columns by binding: positions stop mattering
		}
		cur = parent.get();
	}
}

void AIJoinRewrite::TryPushBelowJoin(unique_ptr<LogicalOperator> &op, unique_ptr<LogicalOperator> &root) {
	if (!LogicalAIRegion::TryCast(*op) || op->children.size() != 1) {
		return;
	}
	auto &region = *LogicalAIRegion::TryCast(*op);
	const auto pre_push_out = op->GetColumnBindings(); // parents' positional maps index THIS order
	// A region carrying a pushed LIMIT k early-stops on the fan-out counts of its INPUT rows; below the join it
	// no longer sees the fan-out, so the count-weighted stop would be wrong. Keep it above the join (Stage 1).
	if (region.limit >= 0) {
		return;
	}
	// The join is either the region's direct child, or reached through ONE projection (AIRegionRewrite hoists
	// the region above the Filter/Projection child chain; a decompress/arg-building projection commonly sits
	// between the region and the join).
	optional_ptr<LogicalProjection> proj;
	LogicalOperator *join_ptr = region.children[0].get();
	if (join_ptr->type == LogicalOperatorType::LOGICAL_PROJECTION && join_ptr->children.size() == 1) {
		proj = &join_ptr->Cast<LogicalProjection>();
		join_ptr = join_ptr->children[0].get();
	}
	const bool is_inner = join_ptr->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
	                      join_ptr->Cast<LogicalComparisonJoin>().join_type == JoinType::INNER;
	const bool is_cross = join_ptr->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT;
	if ((!is_inner && !is_cross) || join_ptr->children.size() != 2) {
		return;
	}
	// The AI call rewritten to read the join's output directly: through a projection, inline the projected
	// expressions into the call; else use it as-is.
	auto inlined = region.expressions[0]->Copy();
	if (proj && !InlineProjectionRefs(inlined, *proj)) {
		return;
	}
	// All columns the call reads must come from exactly ONE join side; that side is where the region goes.
	vector<ColumnBinding> keys;
	CollectColumnRefs(*inlined, keys);
	if (keys.empty()) {
		return; // constant-arg AI call: nothing to key a side on
	}
	const auto left_bindings = join_ptr->children[0]->GetColumnBindings();
	const auto right_bindings = join_ptr->children[1]->GetColumnBindings();
	idx_t side;
	if (AllContainedIn(keys, left_bindings)) {
		side = 0;
	} else if (AllContainedIn(keys, right_bindings)) {
		side = 1;
	} else {
		return; // multi-side key: the AI call spans both inputs -> keep the Stage-1 region above the join
	}
	// Call-optimality gate: below an INNER join, only push onto a side the (purely relational) Yannakakis
	// semi-join reduction has already restricted to surviving keys. Below a cross product every side row
	// reaches the output, so the push is always call-neutral.
	if (is_inner && !SubtreeHasSemiReduction(*join_ptr->children[side])) {
		return;
	}
	const idx_t side_binding_count = side == 0 ? left_bindings.size() : right_bindings.size();
	const auto result_type = region.expressions[0]->GetReturnType();
	const auto dedup_index = region.region_index;

	// --- rebuild ---
	// Region[P?[Join(..)]]  ->  P'?[Join(.., Region[side])]. The region's output binding (dedup_index, 0) flows
	// up through the join (its bindings are the union of its children's); with a projection on top it is
	// re-exposed via an appended passthrough, and consumers are rebound to the projection's new column.
	auto region_node = std::move(op);      // owns Region (child: P or Join)
	unique_ptr<LogicalOperator> proj_node; // owns P if present
	unique_ptr<LogicalOperator> join_node; // owns the Join
	if (proj) {
		proj_node = std::move(region_node->children[0]);
		join_node = std::move(proj_node->children[0]);
	} else {
		join_node = std::move(region_node->children[0]);
	}
	LogicalAIRegion::TryCast(*region_node)->expressions[0] = std::move(inlined);
	region_node->children[0] = std::move(join_node->children[side]);
	join_node->children[side] = std::move(region_node);
	// A non-empty join projection map filters which side columns the join outputs -- the region's appended
	// result column (last position of the side child) must be added or it would be silently dropped.
	if (is_inner) {
		auto &cj = join_node->Cast<LogicalComparisonJoin>();
		auto &map = side == 0 ? cj.left_projection_map : cj.right_projection_map;
		if (!map.empty()) {
			map.push_back(ProjectionIndex(side_binding_count));
		}
	}
	if (proj) {
		auto &proj_ref = proj_node->Cast<LogicalProjection>();
		const idx_t new_col = proj_ref.expressions.size();
		proj_node->children[0] = std::move(join_node);
		op = std::move(proj_node);
		// Rebind consumers of the region result to the projection's about-to-be-appended output. Runs BEFORE
		// the passthrough exists, so the only matches are the true consumers above this projection.
		AIRegionBindingReplacer replacer(ColumnBinding(dedup_index, ProjectionIndex(0)),
		                                 ColumnBinding(proj_ref.table_index, ProjectionIndex(new_col)));
		replacer.VisitOperator(*root);
		op->expressions.push_back(
		    make_uniq<BoundColumnRefExpression>(result_type, ColumnBinding(dedup_index, ProjectionIndex(0))));
	} else {
		op = std::move(join_node);
	}
	op->ResolveOperatorTypes();
	RemapParentPositionalMaps(*root, *op, pre_push_out);
}

void AIJoinRewrite::Rewrite(unique_ptr<LogicalOperator> &op, unique_ptr<LogicalOperator> &root) {
	for (auto &child : op->children) {
		Rewrite(child, root);
	}
	if (factor_mode && TryFactorGraph(op)) {
		return;
	}
	TryMarkExistential(*op);
	TryPushBelowJoin(op, root);
}

//! Remove speculative pre-filters from a factor-graph side: the graph's unary pre-pass prunes the
//! side EXACTLY, so an estimate-gated leaf node below it is redundant machinery (its calls would
//! dedup, but its embeds and MLP work would not). Result-preserving by the speculative contract.
static void StripSpeculativeBelow(unique_ptr<LogicalOperator> &op) {
	for (auto &child : op->children) {
		StripSpeculativeBelow(child);
	}
	if (op->type != LogicalOperatorType::LOGICAL_FILTER) {
		return;
	}
	auto &filter = op->Cast<LogicalFilter>();
	for (idx_t i = filter.expressions.size(); i-- > 0;) {
		auto &expr = *filter.expressions[i];
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION &&
		    expr.Cast<BoundFunctionExpression>().Function().GetName().StartsWith("speculative_ai_")) {
			filter.expressions.erase(filter.expressions.begin() + NumericCast<int64_t>(i));
		}
	}
	if (filter.expressions.empty() && filter.projection_map.empty() && op->children.size() == 1) {
		op = std::move(op->children[0]);
	}
}

//! A MARK join needs only a boolean per RHS key value. When its RHS is (projections over) a
//! factor graph and the mark keys resolve to ONE graph side, tag that side EXISTENTIAL: the
//! scheduler completes each member at its first confirmed tuple and downstream work flows only
//! for unsatisfied members. Sound for mark semantics: >=1 emitted tuple per existing member,
//! none per non-existing.
void AIJoinRewrite::TryMarkExistential(LogicalOperator &op) {
	if (op.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return;
	}
	auto &join = op.Cast<LogicalComparisonJoin>();
	if (join.join_type != JoinType::MARK || join.children.size() != 2) {
		return;
	}
	vector<ColumnBinding> targets;
	for (auto &cond : join.conditions) {
		if (!cond.IsComparison() || cond.GetRHS().GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			return;
		}
		targets.push_back(cond.GetRHS().Cast<BoundColumnRefExpression>().Binding());
	}
	if (targets.empty()) {
		return;
	}
	reference<LogicalOperator> cur = *join.children[1];
	while (cur.get().type == LogicalOperatorType::LOGICAL_PROJECTION && cur.get().children.size() == 1) {
		auto &proj = cur.get().Cast<LogicalProjection>();
		for (auto &b : targets) {
			if (b.table_index != proj.table_index) {
				return; // key does not come through this projection
			}
			auto &expr = proj.expressions[b.column_index.GetIndex()];
			if (expr->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				return;
			}
			b = expr->Cast<BoundColumnRefExpression>().Binding();
		}
		cur = *cur.get().children[0];
	}
	auto *graph = LogicalAIFactorGraph::TryCast(cur.get());
	if (!graph || graph->existential_side != DConstants::INVALID_INDEX) {
		return;
	}
	for (idx_t s = 0; s < graph->children.size(); s++) {
		auto side_bindings = graph->children[s]->GetColumnBindings();
		bool all = true;
		for (auto &b : targets) {
			all = all && std::find(side_bindings.begin(), side_bindings.end(), b) != side_bindings.end();
		}
		if (all) {
			graph->existential_side = s;
			return;
		}
	}
}

bool AIJoinRewrite::TryFactorGraph(unique_ptr<LogicalOperator> &op) {
	// Match: Filter(sole predicate == #region output) over AIRegion(folded conjunctive node) over a
	// chain of cross products. The folded node's leaves must each read one side (unary) or two
	// sides (edge), with at least one edge -- then the region collapses into an n-ary factor graph
	// (member/pair domains with exact backward pruning; the cross product never materializes).
	if (op->type != LogicalOperatorType::LOGICAL_FILTER || op->expressions.size() != 1 || op->children.size() != 1) {
		return false;
	}
	auto *region = LogicalAIRegion::TryCast(*op->children[0]);
	if (!region || region->limit >= 0 || region->children.size() != 1 ||
	    region->children[0]->type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
		return false;
	}
	auto &pred_ref = *op->expressions[0];
	if (pred_ref.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return false;
	}
	auto &colref = pred_ref.Cast<BoundColumnRefExpression>();
	if (colref.Binding().table_index != region->region_index || colref.Binding().column_index.GetIndex() != 0) {
		return false;
	}
	if (region->expressions[0]->GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	// A sole PLAIN AI predicate (the common 2-way sem_filter(a,b)) is folded into a single-leaf
	// ai_function_with_embed on the fly so the graph -- adaptive streaming, LIMIT/EXISTS, and the
	// prefix-cache declaration -- covers it too. The fold is held locally and only committed when
	// every graph check passes; on any rejection the plan is left untouched for the fallbacks.
	unique_ptr<Expression> folded_local;
	{
		auto &plain_fn = region->expressions[0]->Cast<BoundFunctionExpression>();
		if (plain_fn.Function().GetName() != "ai_function_with_embed") {
			if (!AIIsMixedBoolean(*region->expressions[0])) {
				return false; // speculative wrappers and non-AI predicates never fold
			}
			vector<AIMixedLeaf> leaves;
			auto mixed_tree = AIBuildMixedTree(*region->expressions[0], leaves);
			if (!mixed_tree || leaves.empty()) {
				return false;
			}
			const string tree_str = AIFilterTreeSerialize(*mixed_tree);
			vector<unique_ptr<Expression>> call_prompts, feat_pred, feat_input;
			string meta_str;
			if (!AIBuildMixedLeafArgs(optimizer.context, leaves, call_prompts, feat_pred, feat_input, meta_str)) {
				return false; // a leaf was not bakeable -> leave the plan untouched
			}
			vector<unique_ptr<Expression>> args;
			args.reserve(2 + 3 * leaves.size());
			args.push_back(make_uniq<BoundConstantExpression>(Value(tree_str)));
			for (auto &e : call_prompts) {
				args.push_back(std::move(e));
			}
			for (auto &e : feat_pred) {
				args.push_back(std::move(e));
			}
			for (auto &e : feat_input) {
				args.push_back(std::move(e));
			}
			args.push_back(make_uniq<BoundConstantExpression>(Value(meta_str)));
			auto &catalog = Catalog::GetSystemCatalog(optimizer.context);
			auto &entry = catalog.GetEntry<ScalarFunctionCatalogEntry>(
			    optimizer.context,
			    QualifiedName(catalog.GetName(), Identifier::DefaultSchema(), "ai_function_with_embed"));
			FunctionBinder function_binder(optimizer.context);
			ErrorData error;
			folded_local = function_binder.BindScalarFunction(entry, std::move(args), error);
			if (!folded_local) {
				return false;
			}
		}
	}
	auto &fn = (folded_local ? *folded_local : *region->expressions[0]).Cast<BoundFunctionExpression>();
	// the folded tree must be a pure conjunction of leaves (per-leaf falsity kills its tuples)
	if (fn.GetChildren().empty() || fn.GetChildren()[0]->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
		return false;
	}
	auto &tree_val = fn.GetChildren()[0]->Cast<BoundConstantExpression>().GetValue();
	if (tree_val.IsNull()) {
		return false;
	}
	auto tree = AIFilterTreeParse(StringValue::Get(tree_val));
	if (!tree) {
		return false;
	}
	if (tree->type == AIFilterTreeType::AND_OP) {
		for (auto &child : tree->children) {
			if (child->type != AIFilterTreeType::LEAF) {
				return false;
			}
		}
	} else if (tree->type != AIFilterTreeType::LEAF) {
		return false;
	}
	const idx_t n = tree->LeafCount();
	if (fn.GetChildren().size() < 1 + 3 * n) {
		return false;
	}
	// flatten the cross chain into its base sides
	vector<unique_ptr<LogicalOperator> *> sides;
	std::function<void(unique_ptr<LogicalOperator> &)> flatten = [&](unique_ptr<LogicalOperator> &child) {
		if (child->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
			flatten(child->children[0]);
			flatten(child->children[1]);
		} else {
			sides.push_back(&child);
		}
	};
	flatten(region->children[0]);
	if (sides.size() < 2) {
		return false;
	}
	vector<vector<ColumnBinding>> side_bindings;
	side_bindings.reserve(sides.size());
	for (auto *side : sides) {
		side_bindings.push_back((*side)->GetColumnBindings());
	}
	auto side_of_binding = [&](const ColumnBinding &b) -> idx_t {
		for (idx_t s = 0; s < side_bindings.size(); s++) {
			if (std::find(side_bindings[s].begin(), side_bindings[s].end(), b) != side_bindings[s].end()) {
				return s;
			}
		}
		return side_bindings.size();
	};
	bool ok = true;
	bool any_edge = false;
	std::function<void(const Expression &, vector<idx_t> &)> collect_sides = [&](const Expression &e,
	                                                                             vector<idx_t> &leaf_sides) {
		if (e.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
			const idx_t s = side_of_binding(e.Cast<BoundColumnRefExpression>().Binding());
			if (s >= side_bindings.size()) {
				ok = false; // references something outside the cross -> not our shape
			} else if (std::find(leaf_sides.begin(), leaf_sides.end(), s) == leaf_sides.end()) {
				leaf_sides.push_back(s);
			}
			return;
		}
		ExpressionIterator::EnumerateChildren(e, [&](const Expression &child) { collect_sides(child, leaf_sides); });
	};
	for (idx_t l = 0; l < n && ok; l++) {
		vector<idx_t> leaf_sides;
		for (idx_t part = 0; part < 3; part++) {
			collect_sides(*fn.GetChildren()[1 + part * n + l], leaf_sides);
		}
		if (leaf_sides.empty() || leaf_sides.size() > 2) {
			ok = false;
		}
		any_edge = any_edge || leaf_sides.size() == 2;
	}
	if (!ok || !any_edge) {
		return false;
	}
	auto graph = make_uniq<LogicalAIFactorGraph>(
	    region->region_index, folded_local ? std::move(folded_local) : std::move(region->expressions[0]));
	for (auto *side : sides) {
		StripSpeculativeBelow(*side);
		graph->children.push_back(std::move(*side));
	}
	graph->ResolveOperatorTypes();
	op = std::move(graph);
	return true;
}

//===--------------------------------------------------------------------===//
// DISTINCT / duplicate-insensitive aggregate consumers: skip the expand entirely
//===--------------------------------------------------------------------===//

namespace {
// Small binding set (linear -- these sets stay tiny).
struct BindingSet {
	vector<ColumnBinding> v;
	bool Contains(const ColumnBinding &b) const {
		return std::find(v.begin(), v.end(), b) != v.end();
	}
	void Add(const ColumnBinding &b) {
		if (!Contains(b)) {
			v.push_back(b);
		}
	}
	void Remove(const ColumnBinding &b) {
		v.erase(std::remove(v.begin(), v.end(), b), v.end());
	}
	bool Empty() const {
		return v.empty();
	}
};
} // namespace

static void AddColumnRefs(const Expression &e, BindingSet &s) {
	if (e.GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
		s.Add(e.Cast<BoundColumnRefExpression>().Binding());
	}
	ExpressionIterator::EnumerateChildren(e, [&](const Expression &c) { AddColumnRefs(c, s); });
}

// Scope check: an AI call anywhere in the expression (e.g. `GROUP BY ai_classify(x)` holds the call in the
// aggregate's group exprs, where no region is ever built -- the SEMI conversion is then the only lever that
// shrinks the call's input from the fan-out to the survivors).
static bool ExprHasAICall(const Expression &e) {
	if (e.GetExpressionType() == ExpressionType::BOUND_FUNCTION) {
		const auto name = e.Cast<BoundFunctionExpression>().Function().GetName();
		if (name == "ai_function_with_embed" || name == "ai_filter" || name == "ai_classify" || name == "ai_score" ||
		    name == "ai_complete") {
			return true;
		}
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(e, [&](const Expression &c) {
		if (!found && ExprHasAICall(c)) {
			found = true;
		}
	});
	return found;
}

static bool OpHasAICall(const LogicalOperator &op) {
	for (auto &e : op.expressions) {
		if (ExprHasAICall(*e)) {
			return true;
		}
	}
	return false;
}

// A consumer whose result depends only on the SET of its input rows (not the multiset): plain DISTINCT, or an
// aggregate whose every function is duplicate-insensitive (DISTINCT-qualified, min/max, bool_and/bool_or).
// Fills `needed` with every column the consumer reads. DISTINCT ON is row-picking (order-sensitive) -> false.
static bool ConsumerIsDuplicateInsensitive(LogicalOperator &op, BindingSet &needed, bool &has_ai) {
	if (op.type == LogicalOperatorType::LOGICAL_DISTINCT) {
		auto &distinct = op.Cast<LogicalDistinct>();
		if (distinct.distinct_type != DistinctType::DISTINCT) {
			return false;
		}
		for (auto &t : distinct.distinct_targets) {
			AddColumnRefs(*t, needed);
			has_ai = has_ai || ExprHasAICall(*t);
		}
		if (needed.Empty()) {
			for (auto &b : op.children[0]->GetColumnBindings()) {
				needed.Add(b);
			}
		}
		return true;
	}
	if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = op.Cast<LogicalAggregate>();
		if (agg.grouping_sets.size() > 1 || !agg.grouping_functions.empty()) {
			return false;
		}
		for (auto &g : agg.groups) {
			AddColumnRefs(*g, needed);
			has_ai = has_ai || ExprHasAICall(*g);
		}
		for (auto &e : agg.expressions) {
			if (e->GetExpressionType() != ExpressionType::BOUND_AGGREGATE) {
				return false;
			}
			auto &ae = e->Cast<BoundAggregateExpression>();
			const auto name = ae.Function().GetName();
			const bool insensitive =
			    ae.IsDistinct() || name == "min" || name == "max" || name == "bool_and" || name == "bool_or";
			if (!insensitive) {
				return false; // e.g. count(*)/sum: the fan-out multiplicity IS the answer
			}
			for (auto &c : ae.GetChildren()) {
				AddColumnRefs(*c, needed);
				has_ai = has_ai || ExprHasAICall(*c);
			}
			if (ae.GetFilter()) {
				AddColumnRefs(*ae.GetFilter(), needed);
			}
		}
		return true;
	}
	return false;
}

// A duplicate-insensitive consumer reading only ONE side S of an INNER join below it makes the join's fan-out
// meaningless: the distinct SET of S-side tuples in the join output is exactly `S with >= 1 match` = S SEMI O.
// Convert the join to (RIGHT_)SEMI so the fan-out is never produced ANYWHERE -- each S row flows once, the AI
// region (in the chain) folds the survivors, and the expand/broadcast step disappears with the duplicates.
// Intermediate ops are per-row functions/filters over S columns, so they map the same set either way.
bool AIJoinRewrite::TrySemiConvertForConsumer(LogicalOperator &consumer) {
	BindingSet needed;
	bool saw_ai = false;
	if (consumer.children.size() != 1 || !ConsumerIsDuplicateInsensitive(consumer, needed, saw_ai) || needed.Empty()) {
		return false;
	}
	// Walk down through row-preserving ops, tracking what each level needs from below. Projections are opaque
	// (their outputs replace the visible bindings); regions/filters pass child bindings through. A projection
	// expression nobody above needs is recorded for nullification -- after the SEMI conversion the dropped
	// side's bindings no longer exist, so dead expressions over them (e.g. a decompress of the other side's
	// column) must not dangle; replacing them with a typed NULL keeps output arity and indices stable.
	struct DeadExprs {
		LogicalProjection *proj;
		vector<idx_t> dead;
	};
	vector<DeadExprs> deads;
	bool saw_region = false;
	vector<LogicalOperator *> chain; // the row-preserving ops between the consumer and the join, top-down
	LogicalOperator *cur = consumer.children[0].get();
	while (cur->children.size() == 1) {
		if (cur->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &proj = cur->Cast<LogicalProjection>();
			BindingSet below;
			DeadExprs de {&proj, {}};
			for (idx_t i = 0; i < proj.expressions.size(); i++) {
				if (needed.Contains(ColumnBinding(proj.table_index, ProjectionIndex(i)))) {
					AddColumnRefs(*proj.expressions[i], below);
				} else {
					de.dead.push_back(i);
				}
			}
			deads.push_back(std::move(de));
			needed = std::move(below);
		} else if (LogicalAIRegion::TryCast(*cur)) {
			auto &region = *LogicalAIRegion::TryCast(*cur);
			needed.Remove(ColumnBinding(region.region_index, ProjectionIndex(0)));
			for (auto &e : region.expressions) {
				AddColumnRefs(*e, needed); // the region evaluates its call regardless of consumers
			}
			saw_region = true;
		} else if (cur->type == LogicalOperatorType::LOGICAL_FILTER) {
			for (auto &e : cur->expressions) {
				AddColumnRefs(*e, needed);
			}
		} else {
			break;
		}
		saw_ai = saw_ai || OpHasAICall(*cur);
		chain.push_back(cur);
		cur = cur->children[0].get();
	}
	if (cur->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN || cur->children.size() != 2) {
		return false;
	}
	auto &join = cur->Cast<LogicalComparisonJoin>();
	if (join.join_type != JoinType::INNER || needed.Empty() || !(saw_region || saw_ai)) {
		return false; // scoped to AI plans (a region or AI call in the consumer/chain); relational stays native
	}
	const auto left_bindings = join.children[0]->GetColumnBindings();
	const auto right_bindings = join.children[1]->GetColumnBindings();
	idx_t side;
	if (AllContainedIn(needed.v, left_bindings)) {
		side = 0;
	} else if (AllContainedIn(needed.v, right_bindings)) {
		side = 1;
	} else {
		return false; // the consumer genuinely reads both sides -> multiplicity may matter downstream shapes
	}
	// A filter on the chain may carry a POSITIONAL projection_map (ColumnLifetimeAnalyzer) over its child's output.
	// The conversion drops the other side's columns from the join's output, and regions and map-less filters
	// pass that change up, so every such map is remapped by binding identity -- bottom-up, because each op's
	// new output depends on the maps below it. (A stale map selected the region's result in place of the kept
	// column: "Failed to bind column reference", SWAN 2.0 DISTINCT-over-join.)
	vector<vector<ColumnBinding>> old_child_out;
	for (auto *op : chain) {
		old_child_out.push_back(op->children[0]->GetColumnBindings());
	}
	join.join_type = side == 0 ? JoinType::SEMI : JoinType::RIGHT_SEMI;
	for (idx_t i = chain.size(); i-- > 0;) {
		if (chain[i]->type == LogicalOperatorType::LOGICAL_FILTER) {
			auto &map = chain[i]->Cast<LogicalFilter>().projection_map;
			if (!map.empty()) {
				PositionalMapSnapshot::RemapByIdentity(map, old_child_out[i],
				                                       chain[i]->children[0]->GetColumnBindings());
			}
		}
	}
	for (auto &de : deads) {
		for (const idx_t i : de.dead) {
			de.proj->expressions[i] =
			    make_uniq<BoundConstantExpression>(Value(de.proj->expressions[i]->GetReturnType()));
		}
	}
	return true;
}

void AIJoinRewrite::SemiConvertSweep(LogicalOperator &op) {
	for (auto &child : op.children) {
		SemiConvertSweep(*child);
	}
	if (TrySemiConvertForConsumer(op)) {
		semi_converted = true;
	}
}

unique_ptr<LogicalOperator> AIJoinRewrite::Optimize(unique_ptr<LogicalOperator> op) {
	// ai_join_factorize: 'off' keeps the region above the join unchanged; 'pushdown' (and, until
	// Stage 6 lands the factor representation, 'factor' too) enables the push-below + semi-convert.
	const string mode = AIVarcharSetting(optimizer.context, "ai_join_factorize", "factor");
	if (mode == "off") {
		return op;
	}
	factor_mode = mode == "factor";
	// DISTINCT/aggregate consumers first: an INNER converted to SEMI no longer needs (or allows) the region
	// push-below -- the SEMI already never fans out, so the region above it folds survivors with no expand.
	// The push-below moves a region (and its appended result column) under a join side, and the SEMI conversion
	// narrows a join: both shift positions in ancestors' maps. Filter/order maps are fixed where each rewrite
	// happens (RemapParentPositionalMaps, TrySemiConvertForConsumer); ancestor JOIN maps are remapped here.
	const PositionalMapSnapshot maps(*op, /*joins_only=*/true);
	SemiConvertSweep(*op);
	Rewrite(op, op);
	maps.Remap(*op);
	if (semi_converted) {
		op->ResolveOperatorTypes(); // the converted joins' output types shrank to the kept side
	}
	return op;
}

} // namespace duckdb
