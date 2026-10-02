//===----------------------------------------------------------------------===//
//                         DuckDB
//
// optimizer/semantic_filter_pullup.hpp
//
// Semantic filter pull-up: the inverse of filter push-down. DuckDB pushes a cheap predicate as low
// as it legally can; an AI predicate costs an LLM call per distinct input, so this pass lifts it as
// high as it legally can, through every operator a filter on its columns commutes with, and settles
// it above the highest row-reducing operator it crossed. Where it crossed one, a speculative
// pre-filter stays at the original site to prune rows the selectivity model expects to fail.
//===----------------------------------------------------------------------===//

#pragma once

#include "optimizer/ai_filter_tree_build.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/common/unordered_set.hpp"

namespace duckdb {

class SemanticFilterPullup {
public:
	explicit SemanticFilterPullup(Optimizer &optimizer) : optimizer(optimizer) {
	}
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	//! AI predicates travelling up the plan together: the ones extracted from one filter (or one join's
	//! conditions), with the pre-filter to leave at that filter once a reducing operator has been crossed.
	struct PulledGroup {
		vector<unique_ptr<Expression>> exprs;     //! routed to the bindings of the operator they have reached
		vector<unique_ptr<Expression>> originals; //! as written at the origin, for putting them back
		vector<idx_t> origin_positions;           //! where each original sat in the origin filter
		optional_ptr<LogicalFilter> origin;       //! the filter they came from; none for join conditions
		unique_ptr<Expression> speculative;       //! pre-filter for `origin`, installed only on a useful lift
		//! The highest row-removing operator crossed so far, and the predicates as bound just above it:
		//! where the group settles if nothing higher commutes usefully.
		optional_ptr<LogicalOperator> checkpoint;
		vector<unique_ptr<Expression>> checkpoint_exprs;
	};

	Optimizer &optimizer;
	//! Filters this pass created: groups that settle at the same place share one (the fold stage wants
	//! the predicates of one site in one filter).
	unordered_set<LogicalFilter *> settled;

	//! Lift the AI predicates out of the subtree at `op`: those that commute through `op` are returned,
	//! routed to `op`'s output bindings; the rest are settled inside the subtree. `extra_columns_ok` says
	//! whether `op` may grow extra output columns (a projection appending a passthrough column); `is_root`
	//! settles everything below `op`.
	vector<PulledGroup> Lift(unique_ptr<LogicalOperator> &op, bool extra_columns_ok, bool is_root);
	//! Take the AI predicates out of `op` itself (a filter's AI conjuncts, an inner join's AI conditions).
	void Extract(unique_ptr<LogicalOperator> &op, vector<PulledGroup> &groups);
	//! Place a group that cannot go higher: in a filter directly above its checkpoint (an operator in the
	//! subtree of `from`), or back into the filter it came from when nothing it crossed removes rows.
	void Settle(LogicalOperator &from, PulledGroup &group);
	//! Put `exprs` in a filter above the operator in `slot`, sharing a filter this pass already put there.
	void InstallAbove(unique_ptr<LogicalOperator> &slot, vector<unique_ptr<Expression>> &exprs);
	//! Build one bound speculative_ai_function_with_embed over the AND of the pulled leaves (baking each
	//! leaf's scalar-exact call prompt + meta; the node embeds predicate/input internally for the MLP
	//! feature). Reads the leaves' expression pointers, so call it BEFORE the leaf exprs are moved away.
	unique_ptr<Expression> BuildSpeculativeCall(vector<AIMixedLeaf> &leaves);
};

} // namespace duckdb
