//===----------------------------------------------------------------------===//
// ai_functions.hpp — registration of the AI scalar/table functions
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb.hpp"

namespace duckdb {

class ExtensionLoader;

//! Registers ai_filter/ai_classify/ai_score/ai_complete/ai_embed/ai_image/ai_agg, the internal
//! ai_function_with_embed nodes, and the usage/training table functions.
void RegisterAIFunctions(ExtensionLoader &loader);

} // namespace duckdb
