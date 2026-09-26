//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/ai_filter_selectivity.hpp
//
// Selectivity signal for ai_filter reordering: P(ai_filter returns true) for a row, produced by the
// online-trained MLP (see ai_selectivity_model.hpp). Cost is a separate concern (ai_prompt_cost.hpp).
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"

namespace duckdb {

//! Estimate P(ai_filter returns true) for a row, from its prompt embedding. Delegates to the
//! online-trained selectivity MLP. Returns a value in (0, 1). `dim == 0` -> 0.5.
double AIEstimateFilterTrueProb(const float *embedding, idx_t dim);

} // namespace duckdb
