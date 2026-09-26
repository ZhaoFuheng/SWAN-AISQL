//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/ai_prompt_cost.hpp
//
// Per-prompt LLM cost estimate for the ai_filter order planner. Independent of the selectivity
// model -- it only needs the prompt text, not any embedding or learned weights.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/string.hpp"

namespace duckdb {

//! Estimate the LLM cost of evaluating an ai_filter for one row, from its prompt. Heuristic
//! stand-in for a real tokenizer: ~ (bytes / 4) input tokens + 1 output token. Always >= 1.
double AIEstimatePromptCost(const string &prompt);

} // namespace duckdb
