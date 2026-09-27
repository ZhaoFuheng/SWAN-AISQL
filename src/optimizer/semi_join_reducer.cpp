#include "optimizer/semi_join_reducer.hpp"
#include <functional>
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

#include "ai_client.hpp"
#include "ai_settings.hpp"

#include <algorithm>

#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator_deep_copy.hpp"
#include "duckdb/common/enums/join_type.hpp"

#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>

namespace duckdb {

static bool ExpressionHasAIFunction(const Expression &expr) {
	if (expr.GetExpressionType() == ExpressionType::BOUND_FUNCTION) {
		auto &fname = expr.Cast<BoundFunctionExpression>().Function().GetName();
		if (fname == "ai_filter" || fname == "ai_classify" || fname == "ai_score" || fname == "ai_complete") {
			return true;
		}
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!found && ExpressionHasAIFunction(child)) {
			found = true;
		}
	});
	return found;
}

// A leaf is "selective" if its subtree carries a relational selection -- a Filter node or a table-scan with
// pushed-down filters. Semi-joining an AI table by such a neighbor shrinks the AI table (few distinct join
// keys survive), so it's worth reducing the AI table by it even when the neighbor has more rows.
static bool SubtreeHasSelection(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
		return true;
	}
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		for (auto &tf : op.Cast<LogicalGet>().table_filters) {
			(void)tf;
			return true; // a pushed-down table filter -> selective
		}
	}
	for (auto &child : op.children) {
		if (SubtreeHasSelection(*child)) {
			return true;
		}
	}
	return false;
}

bool SemiJoinReducer::OperatorHasAIFunction(const LogicalOperator &op) {
	for (auto &expr : op.expressions) {
		if (ExpressionHasAIFunction(*expr)) {
			return true;
		}
	}
	// Filter pushdown folds an above-join AI predicate into the join as a condition, so look there too.
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		for (auto &cond : op.Cast<LogicalComparisonJoin>().conditions) {
			if (cond.IsComparison()) {
				if (ExpressionHasAIFunction(cond.GetLHS()) || ExpressionHasAIFunction(cond.GetRHS())) {
					return true;
				}
			} else if (ExpressionHasAIFunction(cond.GetJoinExpression())) {
				return true;
			}
		}
	}
	return false;
}

bool SemiJoinReducer::SubtreeHasAIFunction(const LogicalOperator &op) {
	if (OperatorHasAIFunction(op)) {
		return true;
	}
	for (auto &child : op.children) {
		if (SubtreeHasAIFunction(*child)) {
			return true;
		}
	}
	return false;
}

bool SemiJoinReducer::ClusterHasAIFunction(const LogicalOperator &op) const {
	if (OperatorHasAIFunction(op)) {
		return true;
	}
	for (auto &child : op.children) {
		if (IsClusterNode(*child)) {
			// descend through the cluster's own inner-join / cross-product nodes
			if (ClusterHasAIFunction(*child)) {
				return true;
			}
		} else if (check_leaves && SubtreeHasAIFunction(*child)) {
			// an ai_filter pushed down into this leaf still feeds off the joins
			return true;
		}
	}
	return false;
}

bool SemiJoinReducer::IsClusterNode(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return op.Cast<LogicalComparisonJoin>().join_type == JoinType::INNER;
	}
	return op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT;
}

void SemiJoinReducer::GatherCluster(unique_ptr<LogicalOperator> &op,
                                    vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> &leaves,
                                    vector<Edge> &edges) {
	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &join = op->Cast<LogicalComparisonJoin>();
		for (auto &cond : join.conditions) {
			// Non-comparison conditions (e.g. an ai_filter predicate folded into the join) carry no
			// equi-join edge; skip them. Calling GetComparisonType() on them would throw.
			if (!cond.IsComparison() || cond.GetComparisonType() != ExpressionType::COMPARE_EQUAL) {
				continue;
			}
			auto &lhs = cond.GetLHS();
			auto &rhs = cond.GetRHS();
			if (lhs.GetExpressionType() != ExpressionType::BOUND_COLUMN_REF ||
			    rhs.GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
				continue;
			}
			auto &lref = lhs.Cast<BoundColumnRefExpression>();
			auto &rref = rhs.Cast<BoundColumnRefExpression>();
			edges.push_back({lref.Binding(), lref.GetReturnType(), rref.Binding(), rref.GetReturnType()});
		}
	}
	for (auto &child : op->children) {
		GatherChild(child, leaves, edges);
	}
}

//! Classify one child of a cluster: recurse into nested clusters, look PAST a materialised-CTE
//! node into its main query, otherwise take it as a leaf.
//! A materialised CTE is an optimisation barrier: treating the node as an opaque leaf hides the
//! whole join tree beneath it, so no reduction reaches the relations inside and a semantic filter
//! down there runs on its full base table. Only the MAIN QUERY (children[1]) is entered; the CTE
//! BODY (children[0]) is a separate scope that other references also read, so pruning it by this
//! query's keys would be wrong.
void SemiJoinReducer::GatherChild(unique_ptr<LogicalOperator> &child,
                                  vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> &leaves,
                                  vector<Edge> &edges) {
	if (IsClusterNode(*child)) {
		GatherCluster(child, leaves, edges);
		return;
	}
	if (child->type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE && child->children.size() > 1) {
		GatherChild(child->children[1], leaves, edges);
		return;
	}
	leaves.push_back(std::ref(child));
}

//! Strip AI predicates from a COPIED reducer subtree, leaving its relational skeleton.
//! A SEMI join reduces by the KEYS the build side produces, so dropping the build copy's AI
//! predicates yields a SUPERSET of that relation's real rows -- the reduction then prunes only
//! tuples that could not join even against the unfiltered neighbour, which is still
//! result-preserving. It just prunes a little less than the full neighbour would.
//! This is what makes reducing BY an AI-bearing neighbour safe: no LLM predicate is duplicated,
//! so there is no extra cost, no verdict-consistency question, and no AI expression in the copy
//! to re-bind. Returns false if an AI call sits somewhere we cannot strip (a join condition or a
//! projection), in which case the caller skips this reduction.
bool AIStripAIPredicates(unique_ptr<LogicalOperator> &op) {
	for (auto &child : op->children) {
		if (!AIStripAIPredicates(child)) {
			return false;
		}
	}
	if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = op->Cast<LogicalFilter>();
		vector<unique_ptr<Expression>> keep;
		for (auto &expr : filter.expressions) {
			if (!ExpressionHasAIFunction(*expr)) {
				keep.push_back(std::move(expr));
			}
		}
		if (keep.empty()) {
			// Purely semantic filter: neutralise it to TRUE rather than splicing the node out.
			// Removing the node would drop its projection_map and change the copy's output
			// bindings, which makes the caller's binding check fail and silently skips the
			// reduction (observed as spliced=0 on three of Q22's six reductions).
			keep.push_back(make_uniq<BoundConstantExpression>(Value::BOOLEAN(true)));
		}
		filter.expressions = std::move(keep);
		return true;
	}
	// any AI call left outside a filter (join condition, projection) is not strippable
	bool has_ai = false;
	for (auto &expr : op->expressions) {
		if (ExpressionHasAIFunction(*expr)) {
			has_ai = true;
		}
	}
	if (op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		for (auto &cond : op->Cast<LogicalComparisonJoin>().conditions) {
			if (!cond.IsComparison() && ExpressionHasAIFunction(cond.GetJoinExpression())) {
				has_ai = true;
			}
		}
	}
	return !has_ai;
}

void SemiJoinReducer::InsertReducers(vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> &leaves,
                                     const vector<Edge> &edges) {
	if (leaves.size() < 2 || edges.empty()) {
		return;
	}

	// Map each relation's table_index to its leaf, and estimate each leaf's cardinality.
	std::unordered_map<idx_t, idx_t> table_to_leaf;
	vector<idx_t> leaf_card(leaves.size(), 0);
	for (idx_t i = 0; i < leaves.size(); i++) {
		auto &leaf = leaves[i].get();
		for (auto &binding : leaf->GetColumnBindings()) {
			table_to_leaf[binding.table_index.index] = i;
		}
		// Size for the reduction heuristic. For an AI leaf, use the RELATIONAL cardinality from just below the
		// ai_filter: the planner's ai_filter selectivity is a fabricated default that collapses the leaf's
		// estimate (e.g. 50 -> 10) and would otherwise defeat "reduce the larger relation by the smaller
		// neighbor" -- the reduction textbook Yannakakis always does on the join keys.
		LogicalOperator *card_op = leaf.get();
		if (SubtreeHasAIFunction(*leaf)) {
			for (LogicalOperator *cur = leaf.get(); cur->children.size() == 1;) {
				const bool is_ai = cur->type == LogicalOperatorType::LOGICAL_FILTER && OperatorHasAIFunction(*cur);
				cur = cur->children[0].get();
				if (is_ai) {
					card_op = cur; // node just below an ai_filter: keeps relational filters, drops the AI estimate
				}
			}
		}
		leaf_card[i] = card_op->EstimateCardinality(optimizer.context);
	}
	const bool yann_dbg = AIConfig::Get().debug_log.find("yann") != string::npos;
	if (yann_dbg) {
		for (idx_t i = 0; i < leaves.size(); i++) {
			fprintf(stderr, "[yann] leaf %llu card=%llu sel=%d\n", (unsigned long long)i,
			        (unsigned long long)leaf_card[i], SubtreeHasSelection(*leaves[i].get()) ? 1 : 0);
		}
	}

	// Group edges by (reduced leaf i, reducer leaf j). Stored edge: left = i's column, right = j's column.
	std::map<std::pair<idx_t, idx_t>, vector<Edge>> reductions;
	for (auto &e : edges) {
		auto it_left = table_to_leaf.find(e.left.table_index.index);
		auto it_right = table_to_leaf.find(e.right.table_index.index);
		if (it_left == table_to_leaf.end() || it_right == table_to_leaf.end()) {
			continue;
		}
		const idx_t i = it_left->second;
		const idx_t j = it_right->second;
		if (i == j) {
			continue;
		}
		reductions[{i, j}].push_back(e);
		reductions[{j, i}].push_back({e.right, e.right_type, e.left, e.left_type});
	}

	// Apply one reducer: leaves[i] <- leaves[i] SEMI JOIN copy-of-leaves[j] on `redu_edges`. The SEMI join is
	// spliced BELOW leaf i's ai_filter so the LLM predicate runs only on the reduced rows. Returns true if it
	// was actually spliced in (false = a safe skip: non-serializable subplan or a binding mismatch).
	auto apply_reducer = [&](idx_t i, idx_t j, const vector<Edge> &redu_edges) -> bool {
		LogicalOperatorDeepCopy deep_copy(optimizer.binder, nullptr);
		unique_ptr<LogicalOperator> reducer_side;
		try {
			reducer_side = deep_copy.DeepCopy(leaves[j].get()); // copy leaf j (possibly already reduced)
		} catch (...) {
			return false;
		}
		// Keep only the relational skeleton of the reducer side (see StripAIPredicates).
		if (!AIStripAIPredicates(reducer_side)) {
			return false;
		}
		reducer_side->ResolveOperatorTypes();
		auto original_bindings = leaves[j].get()->GetColumnBindings();
		auto copy_bindings = reducer_side->GetColumnBindings();
		if (original_bindings.size() != copy_bindings.size()) {
			return false;
		}
		auto reducer = make_uniq<LogicalComparisonJoin>(JoinType::SEMI);
		vector<unique_ptr<Expression>> rhs_exprs; // copy-side key columns
		vector<unique_ptr<Expression>> lhs_exprs; // leaf i's key, TRANSLATED as we descend
		for (auto &cp : redu_edges) {
			ColumnBinding copy_binding; // map j's original join column (cp.right) to the copy's column
			bool found = false;
			for (idx_t b = 0; b < original_bindings.size(); b++) {
				if (original_bindings[b] == cp.right) {
					copy_binding = copy_bindings[b];
					found = true;
					break;
				}
			}
			if (!found) {
				return false;
			}
			lhs_exprs.push_back(make_uniq<BoundColumnRefExpression>(cp.left_type, cp.left));
			rhs_exprs.push_back(make_uniq<BoundColumnRefExpression>(cp.right_type, copy_binding));
		}
		if (lhs_exprs.empty()) {
			return false;
		}
		// Descend BELOW leaf i's ai_filter(s) so the LLM predicate runs on the reduced rows. Filter-pushdown
		// (earlier pass) sits the ai_filter on the scan, so without this the SEMI join lands above it and the
		// LLM still fires on every row. A COMPUTED join key (e.g. CAST(SPLIT_PART(id,'_',2) AS INT)) lives in
		// a projection above the ai_filter; stopping there would leave the SEMI above the AI and save nothing
		// (the agent_bench Q6-Q9 fingerprint: ~150 calls where ~9 candidates survive). Instead the key is
		// TRANSLATED through each projection -- a colref to output k becomes a copy of the projection's k-th
		// expression, rebound to below-projection columns -- and the SEMI compares that expression directly.
		auto translate_through_projection = [](unique_ptr<Expression> &expr, LogicalProjection &proj) -> bool {
			bool ok = true;
			std::function<void(unique_ptr<Expression> &)> rewrite = [&](unique_ptr<Expression> &e) {
				if (e->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
					auto &colref = e->Cast<BoundColumnRefExpression>();
					if (colref.Binding().table_index == proj.table_index) {
						const idx_t k = colref.Binding().column_index.GetIndex();
						if (k >= proj.expressions.size()) {
							ok = false;
							return;
						}
						e = proj.expressions[k]->Copy();
					}
					return;
				}
				ExpressionIterator::EnumerateChildren(*e, [&](unique_ptr<Expression> &child) { rewrite(child); });
			};
			rewrite(expr);
			return ok;
		};
		auto keys_available = [](const vector<unique_ptr<Expression>> &exprs, const vector<ColumnBinding> &binds) {
			bool all = true;
			for (auto &e : exprs) {
				std::function<void(const Expression &)> walk = [&](const Expression &x) {
					if (x.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
						if (std::find(binds.begin(), binds.end(), x.Cast<BoundColumnRefExpression>().Binding()) ==
						    binds.end()) {
							all = false;
						}
						return;
					}
					ExpressionIterator::EnumerateChildren(x, [&](const Expression &c) { walk(c); });
				};
				walk(*e);
			}
			return all;
		};
		// WHERE the SEMI lands decides whether it is worth anything: above the AI predicate it
		// prunes only rows the LLM has already been paid for. `below_ai` records whether the
		// descent actually got underneath one. Without it a reduction that lands uselessly is
		// indistinguishable in the log from one that works -- which cost two wrong diagnoses of
		// the CTE-boundary failure before it existed.
		bool below_ai = false;
		unique_ptr<LogicalOperator> *target = &leaves[i].get();
		vector<unique_ptr<Expression>> target_exprs;
		for (auto &e : lhs_exprs) {
			target_exprs.push_back(e->Copy());
		}
		for (unique_ptr<LogicalOperator> *cur = &leaves[i].get(); *cur && (*cur)->children.size() == 1;) {
			auto &node = **cur;
			vector<unique_ptr<Expression>> next;
			for (auto &e : lhs_exprs) {
				next.push_back(e->Copy());
			}
			if (node.type == LogicalOperatorType::LOGICAL_PROJECTION) {
				bool ok = true;
				for (auto &e : next) {
					if (!translate_through_projection(e, node.Cast<LogicalProjection>())) {
						ok = false;
						break;
					}
				}
				if (!ok) {
					break;
				}
			}
			if (!keys_available(next, node.children[0]->GetColumnBindings())) {
				break;
			}
			lhs_exprs = std::move(next);
			// An AI PROJECTION (ai_complete/ai_classify/... computed per row, e.g. MMQA q1's director
			// extraction in a CTE) is an LLM stage exactly like an ai_filter: once the keys have translated
			// through it -- its pass-through column IS the join key -- the SEMI belongs underneath, so the
			// projection computes only the rows the join keeps. Landing above it prunes rows already paid for.
			const bool is_ai_stage = (node.type == LogicalOperatorType::LOGICAL_FILTER ||
			                          node.type == LogicalOperatorType::LOGICAL_PROJECTION) &&
			                         OperatorHasAIFunction(node);
			cur = &node.children[0];
			if (is_ai_stage) {
				below_ai = true;
				target = cur;
				target_exprs.clear();
				for (auto &e : lhs_exprs) {
					target_exprs.push_back(e->Copy());
				}
			}
		}
		for (idx_t k = 0; k < target_exprs.size(); k++) {
			reducer->conditions.push_back(
			    JoinCondition(std::move(target_exprs[k]), std::move(rhs_exprs[k]), ExpressionType::COMPARE_EQUAL));
		}
		reducer->children.push_back(std::move(*target));
		reducer->children.push_back(std::move(reducer_side));
		reducer->ResolveOperatorTypes();
		*target = std::move(reducer);
		if (yann_dbg) {
			fprintf(stderr, "[yann]   landed below_ai=%d\n", below_ai ? 1 : 0);
		}
		return true;
	};

	// FIXED-POINT (bottom-up two-pass reduction): apply reducers until none new. A leaf reduced in one pass
	// then CARRIES its reducer's selection, so in a later pass it can reduce a neighbor -- this is how a
	// far-end relational predicate propagates through intermediate tables to shrink the AI table (a multi-hop
	// chain, the case single-pass edge reduction misses). Result-preserving regardless of order: every reducer
	// is a SEMI join that only drops non-joining tuples. `applied` fires each (i,j) at most once, so it's
	// idempotent and terminates (finite pairs, monotonically growing).
	std::set<std::pair<idx_t, idx_t>> applied;
	bool changed = true;
	while (changed) {
		changed = false;
		for (auto &kv : reductions) {
			if (applied.count(kv.first)) {
				continue;
			}
			const idx_t i = kv.first.first;  // reduced (kept) leaf
			const idx_t j = kv.first.second; // reducer (build) leaf
			// NOTE: a reducer side estimated at ZERO rows used to be skipped as a degenerate
			// estimate. That inverts the guard's purpose: an empty neighbour is the case where
			// reduction is worth MOST, since the semi-join eliminates everything downstream. It
			// is also correct either way -- a semi-join only drops rows that join nothing, so a
			// wrong estimate costs a useless semi-join, never a wrong answer. agent_bench Q30:
			// the supplier predicate ran on all 50 suppliers while its join admitted none,
			// because the key subquery estimated 0 and the reduction was skipped.
			// Base heuristic: reduce a larger relation by a strictly smaller neighbor. Plus: reduce ANY leaf by
			// a SELECTIVE neighbor (one that carries -- or, after its own reduction in an earlier pass, has
			// inherited -- a relational selection), even when it has more rows (a filtered dimension has few
			// DISTINCT join keys). This must apply to non-AI intermediate tables too, so a far-end predicate
			// propagates through the middle of a chain to reach the AI table. Yannakakis only runs when the
			// cluster carries an AI function, so this stays scoped to AI queries; and reducing extra tables is
			// result-preserving (a SEMI join drops only non-joining tuples).
			bool do_reduce = leaf_card[j] < leaf_card[i];
			if (!do_reduce && SubtreeHasSelection(*leaves[j].get())) {
				do_reduce = true;
			}
			// Reducing BY an AI-bearing neighbour is allowed: the build copy keeps only that
			// neighbour's RELATIONAL skeleton (StripAIPredicates), so no LLM predicate is duplicated
			// and the reduction stays result-preserving (a semi-join against a superset). Blanket
			// refusal used to cost real pruning once CTE inlining spread AI expressions across the
			// join graph -- agent_bench Q22 showed the reducer WANTING to reduce (card_j=20 <
			// card_i=31) and refusing, which left the 2018 review/tip selectivity stranded.
			// A reduction whose AI calls are not strippable is skipped inside apply_reducer.
			if (yann_dbg) {
				fprintf(stderr, "[yann] reduce leaf %llu <- %llu? card_i=%llu card_j=%llu sel_j=%d -> %s\n",
				        (unsigned long long)i, (unsigned long long)j, (unsigned long long)leaf_card[i],
				        (unsigned long long)leaf_card[j], SubtreeHasSelection(*leaves[j].get()) ? 1 : 0,
				        do_reduce ? "YES" : "no");
			}
			if (!do_reduce) {
				continue; // may become reducible once j is itself reduced -- re-checked next pass
			}
			applied.insert(kv.first); // attempt this pair once
			const bool spliced = apply_reducer(i, j, kv.second);
			if (yann_dbg) {
				fprintf(stderr, "[yann]   apply %llu <- %llu spliced=%d\n", (unsigned long long)i,
				        (unsigned long long)j, spliced ? 1 : 0);
			}
			if (spliced) {
				changed = true; // a leaf changed -> re-scan; its neighbors may now be reducible
			}
		}
	}
}

void SemiJoinReducer::ReduceTree(unique_ptr<LogicalOperator> &op, bool ai_above) {
	// An AI function in this operator makes the whole subtree "below an AI filter".
	const bool below_ai = ai_above || OperatorHasAIFunction(*op);
	if (IsClusterNode(*op)) {
		// Reduce this cluster when an AI filter sits above it, is embedded in its joins, or force is on.
		const bool reduce = force || ai_above || ClusterHasAIFunction(*op);
		vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> leaves;
		vector<Edge> edges;
		GatherCluster(op, leaves, edges);
		for (auto &leaf : leaves) {
			ReduceTree(leaf.get(), below_ai); // handle nested clusters first
		}
		if (reduce) {
			InsertReducers(leaves, edges);
		}
	} else {
		// Not reducing here: just keep descending (this is the normal path for AI-free queries).
		for (auto &child : op->children) {
			ReduceTree(child, below_ai);
		}
	}
}

unique_ptr<LogicalOperator> SemiJoinReducer::Optimize(unique_ptr<LogicalOperator> op) {
	// ai_semi_reduce (default true): reduce joins feeding semantic operators. The pipeline-wide
	// early-out already skips pure-relational plans; ai_debug_semi_reduce_force reduces every
	// cluster regardless of AI presence (A/B benchmarking).
	if (!AIBoolSetting(optimizer.context, "ai_semi_reduce", true)) {
		return op;
	}
	force = AIBoolSetting(optimizer.context, "ai_debug_semi_reduce_force", false);
	// A leaf's own AI stage (an ai_filter pushed down into it, or an AI projection feeding the join) counts
	// regardless of the pull-up flag: the reduction shrinks that leaf's LLM input either way.
	check_leaves = true;
	ReduceTree(op, false);
	return op;
}

} // namespace duckdb
