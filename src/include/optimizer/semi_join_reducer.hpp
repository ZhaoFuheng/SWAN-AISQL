//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/semi_join_reducer.hpp
//
// Yannakakis-style semi-join reduction. Inserts SEMI-join reducers that strip
// dangling tuples from base relations before the main (inner) joins run.
// A semi-join R |x S only removes rows of R that cannot join S, so this is
// always result-preserving. Opt-in via the DUCKDB_YANNAKAKIS environment var.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/unordered_map.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/common/types.hpp"

#include <functional>

namespace duckdb {

//! Reduce a subplan to its RELATIONAL SKELETON: AI predicates in filters are neutralised to TRUE
//! (never spliced out -- removing the node drops its projection_map and shifts output bindings).
//! False when an AI call sits somewhere unstrippable (a join condition, a projection). This is
//! what makes reducing BY an AI-bearing neighbour safe: no LLM predicate is duplicated to serve a
//! relational purpose.
bool AIStripAIPredicates(unique_ptr<LogicalOperator> &op);

//! Whether `expr` / any expression in the plan under `op` calls a VOLATILE function other than the AI
//! functions (nextval, random, uuid, ...). A rewrite that re-evaluates such an expression -- a copied
//! neighbour, an inlined CTE body, a join whose fan-out it rides -- would see different values than the
//! original, so every such rewrite declines. The AI functions are marked volatile too (the model may answer
//! differently) but are exactly what the rewrites are about, and their answers are shared through the cache.
bool AIExpressionHasVolatileNonAI(const Expression &expr);
bool AIPlanHasVolatileNonAI(const LogicalOperator &op);
//! The same over one operator's own expression slots only (no descent into its children).
bool AIOperatorHasVolatileNonAI(const LogicalOperator &op);

//! True iff `expr` holds a non-AI expression that can throw at runtime (a cast that can fail, a fallible function,
//! a subquery); the AI functions' own failures are excluded, as for volatility. A rewrite that evaluates such an
//! expression on rows the query would not have evaluated it on (rows a filter rejects) changes a result into an
//! error.
bool AIExpressionCanThrowNonAI(const Expression &expr);
//! The same over every expression an operator holds.
bool AIOperatorCanThrowNonAI(const LogicalOperator &op);
//! Make a SEMI-join key safe to evaluate on any row: every cast that can throw becomes a TRY cast (the same value
//! where the cast succeeds, NULL, so no match, where the query would have thrown). False when the key holds a
//! fallible function, which has no such form, or an AI call.
bool AIMakeKeyInfallible(ClientContext &context, unique_ptr<Expression> &expr);
//! True iff a throwing non-AI expression sits at or above an AI predicate in `op`'s subtree. A copy of such a
//! subtree with its AI predicates stripped (a reducer, a demand body) evaluates that expression on the rows the
//! predicate would have removed: CAST(txt AS INTEGER) AS k over WHERE ai_filter(..) throws on the 'bad' row the
//! filter rejects. Such a subtree is not copied.
//! `bodies` (the demand reduction's CTE index -> body map) lets the walk follow a CTE reference into the body the
//! caller is about to inline: its predicate counts as below the reference. Without it a reference is opaque,
//! which is right for the reducer's copies (they keep the reference; the body is not stripped).
bool AIPlanThrowsAboveAIPredicate(const LogicalOperator &op,
                                  const unordered_map<idx_t, LogicalOperator *> *bodies = nullptr);

class SemiJoinReducer {
public:
	explicit SemiJoinReducer(Optimizer &optimizer) : optimizer(optimizer) {
	}
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	Optimizer &optimizer;
	//! When set (via DUCKDB_SEMANTIC_PULLUP), also treat an ai_filter pushed down INTO a cluster leaf
	//! as a reason to reduce (the semantic filter still feeds off the joins, before it is pulled up).
	bool check_leaves = true;

	//! One equi-join edge discovered inside an inner-join cluster: a column of one leaf against a column of another.
	//! Either side may be the column under a chain of CASTs the binder added for type coercion (an INTEGER key against
	//! a BIGINT one); then `*_expr` holds that side's full expression (the colref inside carries the binding) and the
	//! reducer's SEMI compares the same expression, so a width mismatch no longer loses the reduction.
	struct Edge {
		ColumnBinding left;
		LogicalType left_type;
		ColumnBinding right;
		LogicalType right_type;
		shared_ptr<Expression> left_expr;  //! null = the bare column
		shared_ptr<Expression> right_expr; //! null = the bare column
	};

	//! Does this operator's expressions or join conditions call a row-wise AI function
	//! (ai_filter/classify/score/prompt)?
	static bool OperatorHasAIFunction(const LogicalOperator &op);
	//! Does any node within this inner-join cluster carry an AI function (e.g. an AI predicate folded into a
	//! join, or -- when check_leaves is set -- pushed down into a leaf's filter)?
	bool ClusterHasAIFunction(const LogicalOperator &op) const;
	//! Recursively scan a leaf subtree for an AI function (used only when check_leaves is set).
	static bool SubtreeHasAIFunction(const LogicalOperator &op);

	//! Classify one cluster child: nested cluster / look past a materialised CTE / leaf.
	void GatherChild(unique_ptr<LogicalOperator> &child,
	                 vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> &leaves, vector<Edge> &edges);
	//! Is `op` part of an inner-join cluster (inner comparison join or cross product)?
	static bool IsClusterNode(const LogicalOperator &op);
	//! Recursively find inner-join clusters; reduce a cluster only when an AI filter sits above it (or forced).
	void ReduceTree(unique_ptr<LogicalOperator> &op, bool ai_above);
	//! Collect a cluster's leaf relations (as mutable slots) and equi-join edges.
	void GatherCluster(unique_ptr<LogicalOperator> &op,
	                   vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> &leaves, vector<Edge> &edges);
	//! Wrap each leaf that is larger than a joined neighbor in a SEMI-join reducer.
	//! `scopes[i]`: the materialised CTEs GatherChild looked past on the way to leaf i (a copy of a leaf is in scope
	//! only where the CTEs it reads are defined).
	void InsertReducers(vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> &leaves, const vector<Edge> &edges,
	                    const vector<vector<idx_t>> &scopes);
	//! While gathering: the materialised CTEs looked past so far, and per leaf the ones enclosing it.
	vector<idx_t> scope_stack;
	vector<vector<idx_t>> leaf_scopes;
};

} // namespace duckdb
