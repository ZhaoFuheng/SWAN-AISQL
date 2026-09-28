//===----------------------------------------------------------------------===//
// aisql_optimizer.cpp — the single post-built-ins optimizer pipeline
//
// Registered as an OptimizerExtension optimize_function: runs after ALL of DuckDB's built-in
// passes (positionally identical to the fork's main insertion point). Pipeline order matches the
// fork; stages land incrementally:
//   [4] SemiJoinReducer + SemanticFilterPullup + cleanup re-run     (join-cluster passes)
//   [2] AIPredicateRewrite (reorder/CASE-WHEN folding)              (this stage)
//   [2] AIEmbedFilter (opt-in)                                      (this stage)
//   [3] AIRegionRewrite                                             (region operator stage)
//   [3/6] AIJoinRewrite (pushdown / factor)                         (region + factor stages)
//   [2] AILimitPushdown (last; region half inert until Stage 3)     (this stage)
//===----------------------------------------------------------------------===//
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "optimizer/aisql_optimizer.hpp"

#include "optimizer/ai_cte_split.hpp"
#include "optimizer/ai_cte_demand.hpp"

#include "ai_client.hpp"
#include "ai_settings.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"

#include "optimizer/ai_embed_filter.hpp"
#include "optimizer/ai_limit_pushdown.hpp"
#include "optimizer/ai_predicate_rewrite.hpp"
#include "optimizer/ai_region_rewrite.hpp"
#include "optimizer/ai_join_rewrite.hpp"
#include "optimizer/semantic_filter_pullup.hpp"
#include "optimizer/semi_join_reducer.hpp"
#include "duckdb/optimizer/build_probe_side_optimizer.hpp"
#include "duckdb/optimizer/column_lifetime_analyzer.hpp"
#include "duckdb/optimizer/remove_unused_columns.hpp"
#include "duckdb/planner/binder.hpp"

namespace duckdb {

//! Cheap pipeline-wide early-out: no AI function anywhere in the plan -> no pass runs, so
//! pure-relational plans are never touched (and never pay a traversal per pass).
static bool ExpressionHasAIFunction(const Expression &expr) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
		auto &fn = expr.Cast<BoundFunctionExpression>();
		const auto &name = fn.Function().GetName();
		if (name.StartsWith("ai_") || name.StartsWith("speculative_ai_")) {
			return true;
		}
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!found) {
			found = ExpressionHasAIFunction(child);
		}
	});
	return found;
}

static bool PlanHasAIFunction(LogicalOperator &op) {
	bool found = false;
	// EnumerateExpressions covers every operator-specific container (aggregate groups, join
	// conditions incl. the ANY-join member and comparison-join expression conditions, ...).
	LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *child) {
		if (!found && *child) {
			found = ExpressionHasAIFunction(**child);
		}
	});
	if (found) {
		return true;
	}
	for (auto &child : op.children) {
		if (PlanHasAIFunction(*child)) {
			return true;
		}
	}
	return false;
}

//! Runs in the PRE-optimize hook: marking only, before RunBuiltInOptimizers, so DuckDB's own
//! cte_inlining pass does the transform at its natural position. (Re-running CTEInlining from
//! the post-built-ins slot instead produces WRONG plans -- by then filter pushdown, join
//! ordering and column pruning have specialized the plan around the materialized-CTE shape, and
//! duplicating the body yields a plan that silently returns nothing: agent_bench Q21 gave 0 rows
//! vs 2,712, while the same query with DuckDB's own NOT MATERIALIZED gave the correct 2,712.)
//!
//! DuckDB materializes a CTE referenced more than once. That seals any ai_filter inside it: the
//! semantic pull-up and the Yannakakis reducer both walk join clusters, and a materialized-CTE
//! operator is not one -- so the predicate runs on the CTE's full input instead of on the rows
//! the surrounding joins leave (agent_bench Q26: ~1,430 calls where the chain leaves 5 parts).
//! For an AI-bearing CTE the usual reason to materialize is inverted: re-running it is nearly
//! free because identical prompts dedup to local-cache hits (measured: an inlined AI CTE
//! referenced twice made 0 extra calls, 442 hits), while NOT inlining costs LLM calls. So mark
//! those CTEs NEVER-materialize and re-run DuckDB's inliner. Relational CTEs are untouched.
static bool MarkAICTEsInline(LogicalOperator &op) {
	bool changed = false;
	if (op.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
		auto &cte = op.Cast<LogicalMaterializedCTE>();
		if (cte.materialize != CTEMaterialize::CTE_MATERIALIZE_NEVER && !op.children.empty() &&
		    PlanHasAIFunction(*op.children[0])) {
			cte.materialize = CTEMaterialize::CTE_MATERIALIZE_NEVER;
			changed = true;
		}
	}
	for (auto &child : op.children) {
		if (MarkAICTEsInline(*child)) {
			changed = true;
		}
	}
	return changed;
}

static void AisqlPreOptimizePlan(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	AINoteQuery(input.context.GetCurrentQuery());
	if (!plan || !PlanHasAIFunction(*plan)) {
		return;
	}
	// AISplitCTEsAtSemanticBoundary (optimizer/ai_cte_split.cpp) is the intended replacement for
	// whole-body inlining: the relational prefix becomes an outer materialized CTE that runs ONCE
	// and the thin remainder -- the predicate over a scan of that prefix -- is inlined into every
	// reference, so pushdown and the reducer still get a real subtree per reference.
	//
	// NOT WIRED IN. It is blocked on ONE thing, and the measurements say exactly which. Splitting
	// PINS the AI predicate above the relational work instead of letting pushdown sink it below
	// the joins, which is worth a great deal (agent_bench Q30: 54 -> 7 calls, matching PLOP). But
	// it also puts a CTE scan between the predicate and its leaf, and the semi-join reducer
	// translates its keys DOWN through projections to a scan -- it cannot translate through a CTE
	// scan, so its reducer never lands (computed_key_reduction: 10 -> 200 calls). Restricting the
	// split to CTEs with >=2 references keeps that test green but discards the Q30 win outright
	// (its CTEs have one reference each) and still costs Q26 426 -> 436.
	//
	// So the split is correct and valuable, and it needs the reducer to be able to reduce a CTE
	// reference before it can be turned on. That is the one remaining piece.
	// Two ways to get a semantic CTE's predicate onto the rows its readers can actually use, and
	// they are ALTERNATIVES rather than a stack.
	//
	//   inline the body   -- each reference gets its own copy, which the ordinary reducer then
	//                        prunes in that reference's own scope. Optimal in calls; pays for the
	//                        relational work once per reference.
	//   reduce by demand  -- keep the body materialized and carry the readers' join keys INTO it,
	//                        so the one shared copy is already pruned.
	//
	// Applying both fights: the demand reduction's SEMI sits between the predicate and its scan,
	// and the ordinary reducer walks single-child chains, so it can no longer descend below the
	// predicate and its reduction lands above it, pruning nothing (computed_key_reduction: 10
	// calls -> 178). So inlining stays the default and demand reduction takes over when it is off.
	if (AIBoolSetting(input.optimizer.context, "ai_inline_ai_ctes", true)) {
		MarkAICTEsInline(*plan); // DuckDB's cte_inlining (a built-in, runs next) performs the inlining
	} else {
		AIReduceCTEsByDemand(input.optimizer.binder, plan);
	}
}

static void AisqlOptimizePlan(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	if (!plan || !PlanHasAIFunction(*plan)) {
		return;
	}
	auto &optimizer = input.optimizer;

	// Yannakakis semi-join reduction: strip dangling tuples relationally before any AI evaluation.
	{
		SemiJoinReducer semi_join_reducer(optimizer);
		plan = semi_join_reducer.Optimize(std::move(plan));
	}

	// Semantic filter pull-up: lift multi-table AI filters above the (reduced) join cluster,
	// leaving a speculative node at the leaf where the fan-out gate allows.
	{
		SemanticFilterPullup semantic_pullup(optimizer);
		plan = semantic_pullup.Optimize(std::move(plan));
	}

	// Cleanup re-run of built-in passes: the reducer/pullup ran after DuckDB's column pruning, so
	// re-tighten column sets and re-pick join build sides for the reshaped plan.
	{
		RemoveUnusedColumns unused(optimizer);
		unused.VisitOperator(plan);
		ColumnLifetimeAnalyzer column_lifetime(optimizer, *plan, true);
		column_lifetime.VisitOperator(*plan);
		BuildProbeSideOptimizer build_probe_side(optimizer.context, *plan);
		build_probe_side.VisitOperator(*plan);
	}

	// Per-row AI-predicate reordering: fold a boolean tree of >=2 AI calls into one
	// ai_function_with_embed node (DP order + MLP selectivity + speculative), incl. CASE WHEN.
	{
		AIPredicateRewrite ai_predicate_rewrite(optimizer);
		plan = ai_predicate_rewrite.Optimize(std::move(plan));
	}

	// Embedding pre-filter (opt-in via ai_debug_embed_filter). After the reorder so it finds no
	// bare ai_filters when the reorder already folded them.
	{
		AIEmbedFilter ai_embed_filter(optimizer);
		plan = ai_embed_filter.Optimize(std::move(plan));
	}

	// AI Region placement (ai_factorize: off/filters/all): fold AI calls into factorized
	// dedup+broadcast regions -- above join fan-outs, and (mode 'all') over any child.
	{
		AIRegionRewrite ai_region_rewrite(optimizer);
		plan = ai_region_rewrite.Optimize(std::move(plan));
	}

	// AI join factorization (ai_join_factorize): push the region below the join onto the
	// semi-reduced side + semi-convert duplicate-insensitive consumers ('pushdown'; the 'factor'
	// representation lands in Stage 6).
	{
		AIJoinRewrite ai_join_rewrite(optimizer);
		plan = ai_join_rewrite.Optimize(std::move(plan));
	}

	// LIMIT push-down into AI evaluation: stop calling once k rows pass. Last, after the nodes
	// (and later the regions) it pushes into exist.
	{
		AILimitPushdown ai_limit_pushdown(optimizer);
		plan = ai_limit_pushdown.Optimize(std::move(plan));
	}
}

void RegisterAisqlOptimizer(DatabaseInstance &db) {
	OptimizerExtension ext;
	ext.pre_optimize_function = AisqlPreOptimizePlan;
	ext.optimize_function = AisqlOptimizePlan;
	OptimizerExtension::Register(DBConfig::GetConfig(db), ext);
}

} // namespace duckdb
