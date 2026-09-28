cmake_minimum_required(VERSION 3.5)

# SWAN AI-SQL: semantic SQL (ai_filter / ai_classify / ai_score / ai_complete / ai_agg / ai_embed / ai_image)
# with a two-currency optimizer + AI region / factor-join executor. Built as a DuckDB extension on the
# extension template; dependencies come from vcpkg.json (none required today -- the LLM client uses DuckDB's
# bundled http-only httplib and yyjson; TLS is terminated by the local cache proxy, see docs/DESIGN.md §5).
set(TARGET_NAME aisql)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD "17" CACHE STRING "C++ standard to enforce")
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
    src/optimizer/ai_region_rewrite.cpp
    src/rep/ai_sql_map_chunk.cpp
    src/exec/physical_ai_region.cpp
    src/exec/ai_leaf_region.cpp
    src/optimizer/semi_join_reducer.cpp
    src/optimizer/semantic_filter_pullup.cpp
    src/optimizer/ai_join_rewrite.cpp
    src/plan/logical_ai_factor_graph.cpp
    src/exec/physical_ai_factor_graph.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} "-warnings" ${EXTENSION_SOURCES})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
