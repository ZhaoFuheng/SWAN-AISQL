#include "optimizer/ai_region_rewrite.hpp"
#include <unordered_set>
#include "duckdb/planner/operator/logical_order.hpp"

#include "ai_settings.hpp"

#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/bound_result_modifier.hpp" // BoundLimitNode / LimitNodeType
#include "plan/logical_ai_region.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"

#include <cstdlib>
#include <set>
#include <utility>

namespace duckdb {

//! A join shape whose output an AI filter should be buffered above. INNER/cross fan an input out into
//! DUPLICATES (dedup removes the repeated AI calls). A SEMI join does not fan out, but a Yannakakis semi-join
//! reducer feeds the AI filter its reduced rows in SMALL hash-join chunks (e.g. 3-4 rows) -- and the AI node
//! fires concurrently only WITHIN a chunk, so those become serial per-chunk waves. Wrapping above the SEMI
//! buffers the whole reduced set into ONE fully-concurrent batch. All are input-deterministic, so
//! dedup+broadcast stays result-identical.
static bool IsJoinCluster(const LogicalOperator &op, bool allow_fanout) {
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		const auto jt = op.Cast<LogicalComparisonJoin>().join_type;
		// SEMI does NOT fan out, so buffering above it is concurrency-only (no call-count change) -> always safe.
		if (jt == JoinType::SEMI || jt == JoinType::RIGHT_SEMI) {
			return true;
		}
		// INNER fans out into duplicates: deduping CHANGES the call count, so it is opt-in.
		return allow_fanout && jt == JoinType::INNER;
	}
	return allow_fanout && op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT;
}

// The dedup needs a fan-out below the AI call. That child may be the join cluster directly, or reached through
// a chain of row-preserving Projections above it -- the planner inserts a Projection between the pulled-up
// Filter and the join. Deduping above such projections stays result-preserving (they don't change which rows
// exist, only their columns), and the AI call's key columns are the projection's output that the dedup sees.
static bool HasJoinClusterBelow(const LogicalOperator &op, bool allow_fanout) {
	if (IsJoinCluster(op, allow_fanout)) {
		return true;
	}
	return op.type == LogicalOperatorType::LOGICAL_PROJECTION && op.children.size() == 1 &&
	       HasJoinClusterBelow(*op.children[0], allow_fanout);
}

static bool IsDedupableAICall(const Expression &expr) {
	if (expr.GetExpressionType() != ExpressionType::BOUND_FUNCTION) {
		return false;
	}
	const auto name = expr.Cast<BoundFunctionExpression>().Function().GetName();
	// ai_function_with_embed: the combined reorder node in a Filter above the join. classify/score/prompt:
	// scalar AI calls in a Projection above the join. Plain ai_filter: a SINGLE ai_filter never forms a reorder
	// tree, so it stays scalar -- but it's still input-deterministic and its calls serialize per-chunk above a
	// join, so it benefits from the buffered concurrent batch too (AIDedupEvalScalar handles it). All are
	// input-deterministic, so dedup+broadcast is result-identical.
	return name == "ai_function_with_embed" || name == "ai_filter" || name == "ai_classify" ||
	       name == "ai_score" || name == "ai_complete";
}

static bool ContainsDedupableAICall(const Expression &expr) {
	if (IsDedupableAICall(expr)) {
		return true;
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!found && ContainsDedupableAICall(child)) {
			found = true;
		}
	});
	return found;
}

// Scan-level factorization (ai_factorize='all'): factorize AI calls over ANY child, not just join
// clusters -- a base table can hold many duplicate inputs, and even distinct inputs arrive fragmented
// into per-chunk batches that under-saturate LLM concurrency. 'filters' limits regions to join fan-outs.

// A scan targeted by a runtime DYNAMIC JOIN FILTER (JoinFilterPushdownOptimizer wires a hash join's build side
// to this Get). A region here would SINK the whole scan before the join build publishes that filter, evaluating
// the AI over every row the filter was about to prune (a 1-call plan becomes a full-table scan of LLM calls).
// Native lazy evaluation above the runtime-filtered scan is strictly better -> never factorize such a subtree.
static bool SubtreeHasDynamicFilterScan(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_GET && op.Cast<LogicalGet>().dynamic_filters) {
		return true;
	}
	for (auto &child : op.children) {
		if (SubtreeHasDynamicFilterScan(*child)) {
			return true;
		}
	}
	return false;
}

bool AIRegionRewrite::HoistAICalls(unique_ptr<Expression> &expr, unique_ptr<LogicalOperator> &child_slot,
                                  int64_t limit_k, bool lazy_guard) {
	if (IsDedupableAICall(*expr)) {
		// Under a LIMIT, only an ai_function_with_embed NODE can early-stop while deduping (the count-weighted
		// stop in AIDedupEvaluate/AIFilterEvaluateBatch). A scalar AI call can't, and a blocking dedup would
		// defeat the limit's streaming early-stop -- so leave the scalar in place to stream to the limit.
		const auto &name = expr->Cast<BoundFunctionExpression>().Function().GetName();
		const bool is_node = name == "ai_function_with_embed" || name == "speculative_ai_function_with_embed";
		if (limit_k >= 0 && !is_node) {
			return false;
		}
		const auto return_type = expr->GetReturnType();
		const auto dedup_index = optimizer.binder.GenerateTableIndex();
		auto dedup = make_uniq<LogicalAIRegion>(dedup_index, std::move(expr));
		dedup->limit = is_node ? limit_k : -1; // count-weighted early-stop only on the node path
		dedup->children.push_back(std::move(child_slot));
		dedup->ResolveOperatorTypes();
		child_slot = std::move(dedup);
		expr = make_uniq<BoundColumnRefExpression>(return_type, ColumnBinding(dedup_index, ProjectionIndex(0)));
		return true;
	}
	// Call-safety on the scan path (lazy_guard): a hoisted region evaluates EVERY distinct input, so never
	// hoist from a position native execution evaluates lazily on a row subset -- factorization must never
	// increase LLM calls. CASE: only the first WHEN condition runs on every row (later WHENs run on unmatched
	// rows, THEN/ELSE on their branch subsets). Conjunctions: the adaptive short-circuit evaluates later arms
	// only on surviving rows. (The join path keeps full recursion: there the fan-out dedup is the dominant win.)
	if (lazy_guard) {
		const auto cls = expr->GetExpressionClass();
		if (cls == ExpressionClass::BOUND_CONJUNCTION) {
			return false;
		}
		if (cls == ExpressionClass::BOUND_CASE) {
			auto &case_checks = expr->Cast<BoundCaseExpression>().CaseChecksMutable();
			if (case_checks.empty()) {
				return false;
			}
			return HoistAICalls(case_checks[0].when_expr, child_slot, limit_k, true);
		}
	}
	bool hoisted = false;
	ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
		if (HoistAICalls(child, child_slot, limit_k, lazy_guard)) {
			hoisted = true;
		}
	});
	return hoisted;
}

// A plain constant `LIMIT k` with offset 0 -> returns k; -1 otherwise. An ORDER BY ... LIMIT is a TOP_N (not
// a LOGICAL_LIMIT) and an ORDER_BY between the limit and the node breaks the projection-only descent, so this
// naturally never fires for ordered limits -- exactly the "ignore ORDER BY ... LIMIT" rule we want.
static int64_t GetConstantLimit(const LogicalOperator &op) {
	if (op.type != LogicalOperatorType::LOGICAL_LIMIT) {
		return -1;
	}
	auto &limit = op.Cast<LogicalLimit>();
	if (limit.limit_val.Type() != LimitNodeType::CONSTANT_VALUE) {
		return -1;
	}
	const bool offset_ok = (limit.offset_val.Type() == LimitNodeType::CONSTANT_VALUE &&
	                        limit.offset_val.GetConstantValue() == 0) ||
	                       limit.offset_val.Type() == LimitNodeType::UNSET;
	return offset_ok ? static_cast<int64_t>(limit.limit_val.GetConstantValue()) : -1;
}

void AIRegionRewrite::Rewrite(unique_ptr<LogicalOperator> &op, int64_t limit_k) {
	// Propagate the enclosing constant LIMIT k down through the limit itself and row-preserving projections --
	// the same path AILimitPushdown descends. Any other operator (join, order-by, aggregate) breaks the gate.
	int64_t children_limit = -1;
	const int64_t here = GetConstantLimit(*op);
	if (here >= 0) {
		children_limit = here;
	} else if (op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		children_limit = limit_k;
	}
	for (auto &child : op->children) {
		Rewrite(child, children_limit);
	}
	// A Filter or Projection whose single child is a join cluster: hoist every AI scalar call it contains --
	// top-level (ai_function_with_embed in a Filter; ai_classify/ai_score/ai_complete in a Projection) OR
	// nested inside a predicate (the ai_classify(x) inside `ai_classify(x)='a'`) -- into LogicalAIRegion ops
	// stacked below, so each runs once per distinct input over the whole join output. The predicate / select
	// list then references the deduplicated result column. (Reorder stays ai_filter-specific; this is the
	// operator handling AI functions in a WHERE / SELECT above a join.)
	const bool is_filter = op->type == LogicalOperatorType::LOGICAL_FILTER;
	const bool is_projection = op->type == LogicalOperatorType::LOGICAL_PROJECTION;
	if ((!is_filter && !is_projection) || op->children.size() != 1) {
		return;
	}
	const bool above_join = HasJoinClusterBelow(*op->children[0], allow_fanout_dedup);
	bool lazy_guard = false;
	if (!above_join) {
		// STAGE 3 (scan-level factorization): the same fold -> evaluate-per-rep -> fan-out applies over ANY
		// child. A base table can hold many duplicate inputs (the fold collapses them globally, across chunks,
		// where the scalar path only dedups within one 2048-row chunk), and the region batches distinct reps to
		// the cardinality floor so LLM concurrency saturates instead of firing per-chunk waves. Guards keep it
		// call-safe: (a) a FILTER is factorized only when EVERY conjunct carries a dedupable AI call -- a mixed
		// relational+AI filter keeps native adaptive short-circuit order (base-column predicates were already
		// pushed into the scan; hoisting the AI part would evaluate rows a remaining relational conjunct
		// discards); (b) lazily-evaluated positions never hoist (lazy_guard in HoistAICalls); (c) the LIMIT
		// deferral applies unchanged (scalars under a constant LIMIT stream to the limit's early-stop).
		if (!allow_fanout_dedup || !scan_regions) {
			return;
		}
		if (SubtreeHasDynamicFilterScan(*op->children[0])) {
			return; // a runtime join filter will prune this input at execution; folding it eagerly wastes calls
		}
		if (is_filter) {
			for (auto &expr : op->expressions) {
				if (!ContainsDedupableAICall(*expr)) {
					return;
				}
			}
		}
		lazy_guard = true;
	}
	// Under a constant LIMIT k, push k INTO the dedup so it early-stops the distinct-input evaluation once the
	// PASSING inputs' fan-out counts sum to >= k (count-weighted) -- this is the one shape where dedup and LIMIT
	// COMPOSE: the dedup already keeps the per-input count for its broadcast. HoistAICalls only does this for an
	// ai_function_with_embed node (which supports the count-weighted stop); a scalar AI call is left in place to
	// stream to the limit's natural early-stop (a blocking scalar dedup would defeat it).
	bool changed = false;
	for (auto &expr : op->expressions) {
		if (HoistAICalls(expr, op->children[0], limit_k, lazy_guard)) {
			changed = true;
		}
	}
	if (changed) {
		op->ResolveOperatorTypes();
	}
}

// Record the (aggregate table_index, column) that each ai_agg reads its list argument from.
static void CollectAiAggListRefs(const Expression &e, std::set<std::pair<idx_t, idx_t>> &refs) {
	if (e.GetExpressionType() == ExpressionType::BOUND_FUNCTION) {
		auto &fn = e.Cast<BoundFunctionExpression>();
		if (fn.Function().GetName() == "ai_agg" && !fn.GetChildren().empty() &&
		    fn.GetChildren()[0]->GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
			const auto &b = fn.GetChildren()[0]->Cast<BoundColumnRefExpression>().Binding();
			refs.emplace(b.table_index.index, b.column_index.GetIndex());
		}
	}
	ExpressionIterator::EnumerateChildren(e, [&](const Expression &c) { CollectAiAggListRefs(c, refs); });
}

static void CollectAiAggListRefsOp(const LogicalOperator &op, std::set<std::pair<idx_t, idx_t>> &refs) {
	for (auto &e : op.expressions) {
		CollectAiAggListRefs(*e, refs);
	}
	for (auto &c : op.children) {
		CollectAiAggListRefsOp(*c, refs);
	}
}

// Set DISTINCT on each list()/array_agg() aggregate whose output an ai_agg consumes, so the LLM sees each
// distinct item once. A join fan-out (or repeated group members) would otherwise feed duplicate documents
// into the one aggregate call -- extra tokens and duplicate-document bias. Changes ai_agg output by design.
static void ApplyAiAggDistinct(LogicalOperator &op, const std::set<std::pair<idx_t, idx_t>> &refs) {
	if (op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		auto &agg = op.Cast<LogicalAggregate>();
		for (idx_t k = 0; k < agg.expressions.size(); k++) {
			if (agg.expressions[k]->GetExpressionType() != ExpressionType::BOUND_AGGREGATE) {
				continue;
			}
			auto &ae = agg.expressions[k]->Cast<BoundAggregateExpression>();
			const auto name = ae.Function().GetName();
			if ((name == "list" || name == "array_agg") && !ae.IsDistinct() &&
			    refs.count({agg.aggregate_index.index, k}) > 0) {
				ae.GetAggregateTypeMutable() = AggregateType::DISTINCT;
			}
		}
	}
	for (auto &c : op.children) {
		ApplyAiAggDistinct(*c, refs);
	}
}

//! Positional-map integrity across the hoist: inserting a region under one side of a join APPENDS
//! its result column to that side's output, SHIFTING every later column's position in ancestor
//! outputs. FILTER/ORDER_BY projection maps are positional, so each is snapshotted (its child's
//! binding order) before the pass and remapped by binding IDENTITY afterwards -- identities
//! survive the reshape, only their sequence changes. (Same disease and cure as the join
//! rewrite's push-below; third shape of it found by the agent_bench workload.)
static void SnapshotPositionalMaps(LogicalOperator &op,
                                   vector<std::pair<LogicalOperator *, vector<ColumnBinding>>> &out) {
	const bool has_map =
	    (op.type == LogicalOperatorType::LOGICAL_FILTER && !op.Cast<LogicalFilter>().projection_map.empty()) ||
	    (op.type == LogicalOperatorType::LOGICAL_ORDER_BY && !op.Cast<LogicalOrder>().projection_map.empty());
	if (has_map && !op.children.empty()) {
		out.emplace_back(&op, op.children[0]->GetColumnBindings());
	}
	for (auto &child : op.children) {
		SnapshotPositionalMaps(*child, out);
	}
}

static void CollectLiveOps(LogicalOperator &op, std::unordered_set<LogicalOperator *> &live) {
	live.insert(&op);
	for (auto &child : op.children) {
		CollectLiveOps(*child, live);
	}
}

static void RemapPositionalMaps(LogicalOperator &root,
                                const vector<std::pair<LogicalOperator *, vector<ColumnBinding>>> &snaps) {
	std::unordered_set<LogicalOperator *> live;
	CollectLiveOps(root, live);
	for (auto &snap : snaps) {
		auto *node = snap.first;
		if (!live.count(node) || node->children.empty()) {
			continue;
		}
		vector<ProjectionIndex> *map = nullptr;
		if (node->type == LogicalOperatorType::LOGICAL_FILTER) {
			map = &node->Cast<LogicalFilter>().projection_map;
		} else if (node->type == LogicalOperatorType::LOGICAL_ORDER_BY) {
			map = &node->Cast<LogicalOrder>().projection_map;
		}
		if (!map || map->empty()) {
			continue;
		}
		const auto new_out = node->children[0]->GetColumnBindings();
		const auto &old_out = snap.second;
		for (auto &m : *map) {
			if (m.GetIndex() >= old_out.size()) {
				continue;
			}
			const auto binding = old_out[m.GetIndex()];
			for (idx_t j = 0; j < new_out.size(); j++) {
				if (new_out[j] == binding) {
					m = ProjectionIndex(j);
					break;
				}
			}
		}
	}
}

unique_ptr<LogicalOperator> AIRegionRewrite::Optimize(unique_ptr<LogicalOperator> op) {
	// SCALAR AI fan-out dedup is ON BY DEFAULT (opt OUT with DUCKDB_AI_DEDUP=off/0). A join fan-out repeats
	// each input row; re-evaluating an ai_filter/classify/score/complete on the same input is pure wasted LLM
	// cost, and the answer depends only on the columns the call reads -> dedup+broadcast is result-preserving.
	// (Blocking, but the LIMIT deferral protects early-stop and a large buffer spills to disk.) The SEMI-reducer
	// buffering runs regardless (concurrency-only). ai_agg is EXCLUDED -- see below.
	const string mode = AIVarcharSetting(optimizer.context, "ai_factorize", "all");
	allow_fanout_dedup = mode != "off";
	scan_regions = mode == "all";
	vector<std::pair<LogicalOperator *, vector<ColumnBinding>>> map_snaps;
	SnapshotPositionalMaps(*op, map_snaps);
	Rewrite(op, -1);
	RemapPositionalMaps(*op, map_snaps);
	// ai_agg(list(x)) -> ai_agg(list(DISTINCT x)) is NOT universally result-preserving: it changes the MULTISET
	// the LLM aggregates (a "how many mention X" task would answer differently). So it stays OPT-IN -- applied
	// only when DUCKDB_AI_DEDUP is EXPLICITLY set (and not off), never by the scalar default.
	if (AIBoolSetting(optimizer.context, "ai_debug_agg_distinct", false)) {
		std::set<std::pair<idx_t, idx_t>> agg_refs;
		CollectAiAggListRefsOp(*op, agg_refs);
		if (!agg_refs.empty()) {
			ApplyAiAggDistinct(*op, agg_refs);
		}
	}
	return op;
}

} // namespace duckdb
