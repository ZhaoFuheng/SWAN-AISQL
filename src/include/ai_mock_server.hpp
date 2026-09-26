//===----------------------------------------------------------------------===//
// ai_mock_server.hpp — in-process deterministic mock LLM backend for tests
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb.hpp"

namespace duckdb {

class ExtensionLoader;

//! Start the mock server (idempotent); points the live AIConfig at it. Returns the port.
int AIMockStart();
//! Stop it and restore the previous endpoints. Returns true if a server was running.
bool AIMockStop();

//! Registers CALL ai_mock_start() / ai_mock_stop().
void RegisterAIMockFunctions(ExtensionLoader &loader);

} // namespace duckdb
