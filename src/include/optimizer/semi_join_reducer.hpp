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
	void InsertReducers(vector<std::reference_wrapper<unique_ptr<LogicalOperator>>> &leaves, const vector<Edge> &edges);
};

} // namespace duckdb
