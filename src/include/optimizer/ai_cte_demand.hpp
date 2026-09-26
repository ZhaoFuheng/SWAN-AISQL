//===----------------------------------------------------------------------===//
//                         DuckDB
//
// optimizer/ai_cte_demand.hpp
//
// Import the pruning that lives OUTSIDE a semantic CTE into its body.
//
// A CTE body is a bare relation: it carries no join edges of its own, so Yannakakis has nothing to
// work with there, and the joins that could prune it sit in the query that reads it. This pass
// carries those keys across the boundary -- each reference contributes the key source it joins
// against, and the body is semi-joined against the UNION of them.
//
// The union is the correctness condition, not conservatism: a row must survive if ANY reference
// could consume it. Reducing by one reference's keys looks like a better optimisation and silently
// drops rows (measured in test/sql/semi_reduce/cte_boundary_design.test).
//
// The union need not be tight. A reference contributes its IMMEDIATE join keys; downstream
// narrowing is recovered by the ordinary reducer once the body is inlined, so no transitive demand
// analysis is required.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

//! Semi-joins each AI-bearing CTE body against the union of its references' join keys. Returns
//! true if the plan changed.
bool AIReduceCTEsByDemand(Binder &binder, unique_ptr<LogicalOperator> &plan);

} // namespace duckdb
