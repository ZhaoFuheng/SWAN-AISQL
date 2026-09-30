//===----------------------------------------------------------------------===//
// ai_topn_pushdown.hpp -- evaluate SELECT-list AI calls after ORDER BY ... LIMIT, not before
//
// `SELECT ai_complete(...), x FROM t ORDER BY y LIMIT k` plans as TOP_N over the SELECT-list projection, so every
// row's AI call runs and the top-N keeps k of them. When no sort key reads an AI result, the top-N can run
// first and the projection -- AI calls included -- over the k rows it keeps:
//
//   TOP_N(keys over #p.*)              PROJECTION p (unchanged, same table index)
//     PROJECTION p (.., ai_x(..), ..)   =>  TOP_N(keys with p's plain expressions substituted)
//       child                                child
//
// Result-preserving: the projection is row-wise, so it commutes with keeping k rows, and the substituted keys
// are the same plain, deterministic expressions. Consumers above keep reading p's bindings. Runs first in the
// pipeline, so region placement and join rewrites see the AI projection above a k-row input. Gated by
// ai_limit (LIMIT push-down into AI evaluation).
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class AITopNPushdown {
public:
	explicit AITopNPushdown(Optimizer &optimizer) : optimizer(optimizer) {
	}
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	Optimizer &optimizer;
	void Visit(unique_ptr<LogicalOperator> &op);
};

} // namespace duckdb
