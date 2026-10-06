cmake_minimum_required(VERSION 3.5)

# SWAN AI-SQL: semantic SQL (ai_filter / ai_classify / ai_score / ai_complete /
# ai_agg / ai_embed / ai_image) with a two-currency optimizer + AI region /
# factor-join executor. Built as a DuckDB extension on the extension template;
# the only dependency is OpenSSL (vcpkg.json; optional for local builds, see
# below) for the LLM client, which is DuckDB's bundled httplib plus yyjson.
set(TARGET_NAME aisql)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD
    "17"
    CACHE STRING "C++ standard to enforce")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src/include)

set(EXTENSION_SOURCES
    src/aisql_extension.cpp
    src/client/ai_client.cpp
    src/model/ai_prompt_cost.cpp
    src/model/ai_filter_selectivity.cpp
    src/model/ai_selectivity_model.cpp
    src/model/filter_tree_order.cpp
    src/mock/ai_mock_server.cpp
    src/functions/ai_functions.cpp
    src/plan/logical_ai_region.cpp
    src/optimizer/aisql_optimizer.cpp
    src/optimizer/ai_cte_split.cpp
    src/optimizer/ai_cte_demand.cpp
    src/optimizer/ai_predicate_rewrite.cpp
    src/optimizer/ai_filter_tree_build.cpp
    src/optimizer/ai_limit_pushdown.cpp
    src/optimizer/ai_topn_pushdown.cpp
    src/optimizer/ai_region_rewrite.cpp
    src/rep/ai_sql_map_chunk.cpp
    src/exec/physical_ai_region.cpp
    src/exec/ai_async_dispatcher.cpp
    src/exec/ai_leaf_region.cpp
    src/optimizer/semi_join_reducer.cpp
    src/optimizer/semantic_filter_pullup.cpp
    src/optimizer/ai_join_rewrite.cpp
    src/plan/logical_ai_factor_graph.cpp
    src/exec/physical_ai_factor_graph.cpp)

# TLS: with OpenSSL available (vcpkg.json lists it for CI; Homebrew's openssl@3
# or the system library locally) the LLM client speaks https directly
# (src/include/ai_httplib.hpp). Without it the build is http-only and https
# endpoints go through the local proxies in serve/.
set(OPENSSL_USE_STATIC_LIBS TRUE)
if(EMSCRIPTEN)
  set(OPENSSL_FOUND FALSE) # no sockets in WebAssembly anyway
elseif(
  APPLE
  AND NOT DEFINED OPENSSL_ROOT_DIR
  AND NOT DEFINED ENV{OPENSSL_ROOT_DIR})
  foreach(candidate /opt/homebrew/opt/openssl@3 /usr/local/opt/openssl@3)
    if(EXISTS "${candidate}/include/openssl/ssl.h")
      set(OPENSSL_ROOT_DIR "${candidate}")
      break()
    endif()
  endforeach()
endif()
if(NOT EMSCRIPTEN)
  find_package(OpenSSL)
endif()
if(OPENSSL_FOUND)
  message(STATUS "aisql: TLS client enabled (OpenSSL ${OPENSSL_VERSION})")
  add_compile_definitions(AISQL_TLS=1)
  include_directories(${OPENSSL_INCLUDE_DIR})
else()
  message(
    STATUS
      "aisql: OpenSSL not found, http-only client (https endpoints need the local proxy)"
  )
endif()

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} "-warnings" ${EXTENSION_SOURCES})

if(OPENSSL_FOUND)
  target_link_libraries(${LOADABLE_EXTENSION_NAME} ${OPENSSL_LIBRARIES})
  target_link_libraries(${EXTENSION_NAME} ${OPENSSL_LIBRARIES})
  if(APPLE)
    # httplib verifies against the macOS keychain's root certificates
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
    find_library(SECURITY_FRAMEWORK Security)
    target_link_libraries(${LOADABLE_EXTENSION_NAME}
                          ${COREFOUNDATION_FRAMEWORK} ${SECURITY_FRAMEWORK})
    target_link_libraries(${EXTENSION_NAME} ${COREFOUNDATION_FRAMEWORK}
                          ${SECURITY_FRAMEWORK})
  endif()
endif()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
