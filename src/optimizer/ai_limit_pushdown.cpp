#include "optimizer/ai_limit_pushdown.hpp"

#include "plan/logical_ai_factor_graph.hpp"

#include "ai_settings.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/common/error_data.hpp"
#include "ai_dedup.hpp" // AISetFilterLimit
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/function/function_binder.hpp"
#include "optimizer/ai_filter_tree_build.hpp"
#include "duckdb/planner/bound_result_modifier.hpp" // BoundLimitNode / LimitNodeType
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "plan/logical_ai_region.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"

#include "duckdb/planner/expression_iterator.hpp"

#include <cstdlib>
#include <functional>

namespace duckdb {

static bool IsAIFilterWithEmbed(const Expression &e) {
	return e.GetExpressionType() == ExpressionType::BOUND_FUNCTION &&
	       e.Cast<BoundFunctionExpression>().Function().GetName() == "ai_function_with_embed";
}

unique_ptr<Expression> AILimitPushdown::BuildNode(vector<unique_ptr<Expression>> args) {
	auto &context = optimizer.context;
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto &entry = catalog.GetEntry<ScalarFunctionCatalogEntry>(
	    context, QualifiedName(catalog.GetName(), Identifier::DefaultSchema(), "ai_function_with_embed"));
	FunctionBinder function_binder(context);
	ErrorData error;
	return function_binder.BindScalarFunction(entry, std::move(args), error);
}

//! After a LIMIT k lands in an evaluation above, eager speculative leaf work below it cannot pay
//! off (the k-bounded evaluation above is strictly cheaper): mark every speculative leaf node in
//! the subtree to stand down (pass rows through with no embeds, no calls).
static void StandDownSpeculativeBelow(LogicalOperator &op) {
	LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *child) {
		if (*child) {
			AISetSpeculativeStandDown(**child);
		}
	});
	for (auto &c : op.children) {
		StandDownSpeculativeBelow(*c);
	}
}

bool AILimitPushdown::ApplyLimit(LogicalOperator &filter_op, int64_t k) {
	auto &filter = filter_op.Cast<LogicalFilter>();
	auto &context = optimizer.context;

	// The node's early-stop counts rows that pass the node's whole tree, so it equals the filter result
	// ONLY when the node decides the entire filter. If any other predicate sits above the node it can cut
	// the surviving rows below k -> correctness requires the node to be the SOLE filter predicate. (Base-
	// column relational predicates are already at the scan by now; FILTER_PUSHDOWN ran long before us.)

	// Case 1: an ai_function_with_embed node already exists (e.g. the reorder rewrite built one).
	for (auto &expr : filter.expressions) {
		if (IsAIFilterWithEmbed(*expr)) {
			if (filter.expressions.size() == 1) {
				AISetFilterLimit(*expr, k);
				StandDownSpeculativeBelow(*filter.children[0]);
				return true;
			}
			return false; // other predicates filter above the node -> pushing the limit could return < k
		}
	}

	// Case 2: build one combined ai_function_with_embed node covering EVERY predicate. A single non-AI
	// conjunct would filter above the early-stop -> bail rather than risk returning < k.
	vector<Expression *> exprs;
	for (auto &expr : filter.expressions) {
		exprs.push_back(expr.get());
	}
	auto node = BuildCombinedNode(exprs);
	if (!node) {
		return false; // a non-AI conjunct / unbakeable leaf / bind failure: leave the plan untouched
	}
	AISetFilterLimit(*node, k);
	filter.expressions.clear();
	filter.expressions.push_back(std::move(node));
	filter_op.ResolveOperatorTypes();
	StandDownSpeculativeBelow(*filter.children[0]);
	return true;
}

// One combined ai_function_with_embed over the given boolean predicates (ANDed) -- the same construction the
// reorder rewrite (ai_predicate_rewrite) uses. Each predicate must be a convertible AI boolean tree
// (ai_filter / ai_classify =,IN / ai_score compare / ai_complete =); a lone leaf still benefits (the node's
// worker pool is what supports early termination). Null if any predicate is not convertible or binding fails.
unique_ptr<Expression> AILimitPushdown::BuildCombinedNode(const vector<Expression *> &exprs) {
	auto &context = optimizer.context;
	idx_t leaves = 0;
	for (auto *expr : exprs) {
		if (!AIIsMixedBoolean(*expr)) {
			return nullptr;
		}
		leaves += AICountMixedLeaves(*expr);
	}
	if (leaves == 0) {
		return nullptr;
	}
	vector<AIMixedLeaf> leaf_list;
	vector<unique_ptr<AIFilterTreeNode>> subtrees;
	for (auto *expr : exprs) {
		subtrees.push_back(AIBuildMixedTree(*expr, leaf_list));
	}
	unique_ptr<AIFilterTreeNode> tree = subtrees.size() == 1
	                                        ? std::move(subtrees[0])
	                                        : AIFilterTreeNode::Op(AIFilterTreeType::AND_OP, std::move(subtrees));
	const string tree_str = AIFilterTreeSerialize(*tree);
	const idx_t m = leaf_list.size();

	// Per leaf: the scalar-exact call prompt, the (feat_pred, feat_input) MLP split, and the joined meta_str.
	vector<unique_ptr<Expression>> call_prompts, feat_pred, feat_input;
	string meta_str;
	if (!AIBuildMixedLeafArgs(context, leaf_list, call_prompts, feat_pred, feat_input, meta_str)) {
		return nullptr; // a leaf was not bakeable
	}

	// ai_function_with_embed(tree_str, call_0..m-1, pred_0..m-1, input_0..m-1, meta_str).
	vector<unique_ptr<Expression>> args;
	args.reserve(2 + 3 * m);
	args.push_back(make_uniq<BoundConstantExpression>(Value(tree_str)));
	for (idx_t j = 0; j < m; j++) {
		args.push_back(std::move(call_prompts[j]));
	}
	for (idx_t j = 0; j < m; j++) {
		args.push_back(std::move(feat_pred[j]));
	}
	for (idx_t j = 0; j < m; j++) {
		args.push_back(std::move(feat_input[j]));
	}
	args.push_back(make_uniq<BoundConstantExpression>(Value(meta_str)));
	return BuildNode(std::move(args));
}

// Stage-3 shape: the filter's AI predicate was hoisted into an AI region below it (Filter[colref] ->
// Region[call]). Push k into the REGION instead: its count-weighted Sink early-stop halts evaluation once the
// passing reps' buffered rows sum to >= k, broadcasting the undecided rest as false -- the filter above then
// keeps exactly the first k passers (the same mechanism as the above-join region + limit compose). A scalar
// call is first converted to the combined node (only the node path supports the early-stop).
bool AILimitPushdown::ApplyLimitToRegion(LogicalOperator &filter_op, int64_t k) {
	auto &filter = filter_op.Cast<LogicalFilter>();
	if (filter.expressions.size() != 1 || filter.children.size() != 1) {
		return false;
	}
	auto *region_ptr = LogicalAIRegion::TryCast(*filter.children[0]);
	if (!region_ptr) {
		return false;
	}
	auto &region = *region_ptr;
	if (region.limit >= 0 || region.expressions.empty()) {
		return false;
	}
	// The filter's sole predicate must be exactly the region's appended result column.
	auto &pred = *filter.expressions[0];
	if (pred.GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	auto &binding = pred.Cast<BoundColumnRefExpression>().Binding();
	if (binding.table_index.index != region.region_index.index || binding.column_index.GetIndex() != 0) {
		return false;
	}
	auto &call = region.expressions[0];
	if (IsAIFilterWithEmbed(*call)) {
		AISetFilterLimit(*call, k);
		StandDownSpeculativeBelow(*region.children[0]);
		region.limit = k;
		return true;
	}
	auto node = BuildCombinedNode({call.get()});
	if (!node) {
		return false;
	}
	AISetFilterLimit(*node, k);
	region.expressions[0] = std::move(node); // BOOLEAN, same as the scalar it replaces -> types unchanged
	region.limit = k;
	return true;
}

bool AILimitPushdown::TryPushBelowPreservingJoin(LogicalOperator &join_op, int64_t k) {
	if (join_op.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
	    join_op.type != LogicalOperatorType::LOGICAL_ANY_JOIN) {
		return false;
	}
	// A one-sided outer join drives its output purely from the preserved side (each such row yields >=1
	// output row, in preserved-side order), so early-stopping that side after k passers keeps the same
	// first-k output rows. LEFT/SINGLE preserve the left child, RIGHT the right. FULL OUTER is excluded:
	// its other-side unmatched rows are independent of the AI filter (result-preservation risk), and a
	// WHERE ai_filter(a) over a FULL OUTER stays ABOVE the join anyway (handled by the above-join path).
	// Inner/semi/anti/mark preserve no side (a passer can produce 0 output rows).
	const JoinType jt = join_op.Cast<LogicalJoin>().join_type;
	vector<idx_t> preserved;
	if (jt == JoinType::LEFT || jt == JoinType::SINGLE) {
		preserved.push_back(0);
	} else if (jt == JoinType::RIGHT) {
		preserved.push_back(1);
	} else {
		return false;
	}
	// Descend the preserved side through row-preserving projections to its AI filter and push the limit
	// there. ApplyLimit still requires the node to be that filter's sole predicate, and reaching the join
	// through projections only (in Visit) guarantees nothing filters the join output above the node.
	for (const idx_t ci : preserved) {
		if (ci >= join_op.children.size()) {
			continue;
		}
		reference<LogicalOperator> cur = *join_op.children[ci];
		while (cur.get().type == LogicalOperatorType::LOGICAL_PROJECTION && cur.get().children.size() == 1) {
			cur = *cur.get().children[0];
		}
		if (cur.get().type == LogicalOperatorType::LOGICAL_FILTER &&
		    (ApplyLimit(cur.get(), k) || ApplyLimitToRegion(cur.get(), k))) {
			return true;
		}
	}
	return false;
}

//! Does any expression in this subtree call an AI function (scalar or folded node)?
static bool SubtreeHasAICall(const LogicalOperator &op) {
	for (auto &expr : op.expressions) {
		bool found = false;
		std::function<void(const Expression &)> walk = [&](const Expression &e) {
			if (e.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
				const auto &name = e.Cast<BoundFunctionExpression>().Function().GetName();
				if (name == "ai_filter" || name == "ai_classify" || name == "ai_score" || name == "ai_complete" ||
				    name == "ai_function_with_embed") {
					found = true;
					return;
				}
			}
			ExpressionIterator::EnumerateChildren(e, [&](const Expression &c) { walk(c); });
		};
		walk(*expr);
		if (found) {
			return true;
		}
	}
	if (LogicalAIRegion::Is(op) || LogicalAIFactorGraph::Is(op)) {
		return true;
	}
	for (auto &child : op.children) {
		if (SubtreeHasAICall(*child)) {
			return true;
		}
	}
	return false;
}

//! Cap the AI-bearing side of a cross product with its own LIMIT k. Only the side that actually
//! carries an AI call is capped: this is an AI-aware rewrite, and capping a plain relation would
//! change which rows an unordered LIMIT happens to return for no benefit.
bool AILimitPushdown::TryPushBelowCrossProduct(LogicalOperator &cross_op, int64_t k) {
	if (cross_op.children.size() != 2) {
		return false;
	}
	bool pushed = false;
	for (idx_t ci = 0; ci < 2; ci++) {
		if (!SubtreeHasAICall(*cross_op.children[ci])) {
			continue;
		}
		// The limit has to land BELOW the AI region, not above it. The region is a blocking
		// operator: it consumes its whole input, folds it to distinct values and evaluates the
		// model before emitting a single row, so a LIMIT above it stops nothing -- the calls are
		// already paid for. Capping its INPUT is what makes the scan stop early.
		//
		// Sound only while the path down to the region is projections: the region then emits one
		// row per input row, so k inputs yield k outputs, and k outputs on one side of a cross
		// product yield at least k output rows whenever the other side is non-empty (and when it
		// is empty the result is empty either way). A FILTER anywhere on that path would break it
		// -- k inputs could leave fewer than k survivors -- so the walk stops at anything else.
		reference<unique_ptr<LogicalOperator>> cur = cross_op.children[ci];
		while (cur.get()->type == LogicalOperatorType::LOGICAL_PROJECTION && cur.get()->children.size() == 1) {
			cur = cur.get()->children[0];
		}
		auto *region = LogicalAIRegion::TryCast(*cur.get());
		if (!region || region->children.size() != 1 ||
		    region->children[0]->type == LogicalOperatorType::LOGICAL_LIMIT) {
			continue;
		}
		auto capped = make_uniq<LogicalLimit>(BoundLimitNode::ConstantValue(k), BoundLimitNode());
		capped->children.push_back(std::move(region->children[0]));
		capped->ResolveOperatorTypes();
		region->children[0] = std::move(capped);
		pushed = true;
	}
	return pushed;
}

void AILimitPushdown::Visit(unique_ptr<LogicalOperator> &op) {
	for (auto &child : op->children) {
		Visit(child);
	}
	if (op->type != LogicalOperatorType::LOGICAL_LIMIT || op->children.size() != 1) {
		return;
	}
	auto &limit = op->Cast<LogicalLimit>();
	if (limit.limit_val.Type() != LimitNodeType::CONSTANT_VALUE) {
		return; // only a constant LIMIT k (not a percentage / expression)
	}
	// An OFFSET would need offset+k passers before we could stop; only handle offset 0 / unset for v1.
	if (limit.offset_val.Type() == LimitNodeType::CONSTANT_VALUE) {
		if (limit.offset_val.GetConstantValue() != 0) {
			return;
		}
	} else if (limit.offset_val.Type() != LimitNodeType::UNSET) {
		return;
	}
	const int64_t k = static_cast<int64_t>(limit.limit_val.GetConstantValue());
	if (k <= 0) {
		return;
	}
	// Descend through row-preserving projections to the filter the LIMIT gates.
	reference<LogicalOperator> cur = *op->children[0];
	while (cur.get().type == LogicalOperatorType::LOGICAL_PROJECTION && cur.get().children.size() == 1) {
		cur = *cur.get().children[0];
	}
	if (auto *graph = LogicalAIFactorGraph::TryCast(cur.get())) {
		// The factor graph replaced the sole-predicate filter over the region, so the sole-predicate
		// soundness condition is inherent to the shape: only survivors are emitted. Push k so the
		// lazy/adaptive scheduler stops once k output rows are confirmed.
		if (graph->limit < 0) {
			graph->limit = k;
		}
	} else if (cur.get().type == LogicalOperatorType::LOGICAL_FILTER) {
		ApplyLimit(cur.get(), k);
	} else if (cur.get().type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
		// A CROSS PRODUCT has no join condition, so every row of one side pairs with every row of the
		// other: the first k output rows need at most k rows from either side. Capping the AI-bearing
		// side with its own LIMIT k is therefore result-valid -- it can only lose output rows the
		// outer LIMIT was going to discard -- and it stops the evaluation at the SCAN, which is the
		// only thing that helps when the AI call sits in a PROJECTION rather than a filter (there is
		// no predicate to early-stop on). agent_bench Q1 classifies all 200 books to return 5 rows.
		TryPushBelowCrossProduct(cur.get(), k);
	} else if (cur.get().type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	           cur.get().type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		// LIMIT directly above a join (no filtering projection in between): push k into an AI filter on the
		// join's preserved side, where k passers still yield >=k join-output rows.
		TryPushBelowPreservingJoin(cur.get(), k);
	}
}

unique_ptr<LogicalOperator> AILimitPushdown::Optimize(unique_ptr<LogicalOperator> op) {
	if (!AIBoolSetting(optimizer.context, "ai_limit", true)) {
		return op;
	}
	Visit(op);
	return op;
}

} // namespace duckdb
