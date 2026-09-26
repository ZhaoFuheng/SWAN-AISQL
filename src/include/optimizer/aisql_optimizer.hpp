//===----------------------------------------------------------------------===//
// optimizer/aisql_optimizer.hpp — registration of the aisql optimizer pipeline
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb.hpp"

namespace duckdb {

//! Registers the single post-built-ins OptimizerExtension running the aisql pass pipeline.
void RegisterAisqlOptimizer(DatabaseInstance &db);

} // namespace duckdb
