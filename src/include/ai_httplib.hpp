//===----------------------------------------------------------------------===//
// SWAN AI-SQL: the one include point for DuckDB's bundled cpp-httplib.
//
// With OpenSSL available at build time (vcpkg in CI, Homebrew or the system locally) CMake defines AISQL_TLS and
// the LLM client speaks https directly: the published wheels and CI binaries are such builds. Without it the
// client is http-only and an https endpoint must go through a local proxy (serve/ai_cache_server.py or
// litellm), which is how every benchmark in the repository runs. httplib changes its namespace with TLS on,
// so every user of it goes through the alias below.
//===----------------------------------------------------------------------===//
#pragma once

#ifdef AISQL_TLS
#define CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#include "httplib.hpp"

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
namespace aisql_http = duckdb_httplib_openssl;
#else
namespace aisql_http = duckdb_httplib;
#endif

namespace duckdb {
//! True when this build can open https endpoints itself.
inline bool AIClientHasTLS() {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
	return true;
#else
	return false;
#endif
}
} // namespace duckdb
