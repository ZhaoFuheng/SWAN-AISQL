//===----------------------------------------------------------------------===//
//                         DuckDB
//
// optimizer/ai_lateral_unnest.hpp
//
// Undo the self-join DuckDB's decorrelation builds for `FROM t, LATERAL UNNEST(f(t.x))`.
//
// The planner decorrelates the lateral into two CTEs: a domain D (t's rows) and its duplicate-eliminated
// columns DD, joined back as D ⋈ unnest(f(DD)) on IS NOT DISTINCT FROM. DuckDB's UnnestRewriter turns that
// back into a plain UNNEST over D only when f is a bare column. For a computed argument the self-join stays,
// and once an AI CTE is inlined, D's body -- and its AI predicate -- is copied into DD, where no join key
// reaches it: the copy asks the LLM about every row the query would otherwise prune.
//
// Each D row matches exactly one DD row (DD groups D on the join columns), so the join is
// unnest(f(D's columns)) over D, which is what this pass builds. It changes no result.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

//! Rewrites each decorrelated `D ⋈ unnest(f(DD))` into an UNNEST over D. Runs in the PRE-OPTIMIZE hook, while
//! the planner's shape is intact. Returns true if the plan changed.
bool AIRewriteLateralUnnests(Binder &binder, unique_ptr<LogicalOperator> &plan);

} // namespace duckdb
