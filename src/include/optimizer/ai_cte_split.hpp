//===----------------------------------------------------------------------===//
//                         DuckDB
//
// optimizer/ai_cte_split.hpp
//
// Split an AI-bearing CTE at the semantic boundary: the RELATIONAL body stays materialized and
// runs once, and the SEMANTIC predicate moves out to each reference, where that reference's own
// predicates and joins can prune before any LLM call.
//
// Why this and not inlining the whole body: inlining copies the body per reference, so the
// relational scan runs N times to buy the pushdown (measured on agent_bench Q23: identical calls
// and rows, 10.7s vs 6.1s, 611 cache probes vs 8). Splitting keeps the pushdown -- the predicate
// is at the reference, where the pushdown happens -- and pays for the scan once.
//
// Calls are unchanged by the move: references evaluate the same distinct prompts, and duplicates
// across references are one call anyway (local cache + in-flight registry).
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/binder.hpp"

namespace duckdb {

//! Moves AI predicates out of materialized CTE bodies onto their references. Runs in the
//! PRE-OPTIMIZE hook so DuckDB's own filter pushdown afterwards pushes each reference's
//! relational predicates BELOW the moved predicate. Returns true if the plan changed.
bool AISplitCTEsAtSemanticBoundary(Binder &binder, unique_ptr<LogicalOperator> &plan);

} // namespace duckdb
