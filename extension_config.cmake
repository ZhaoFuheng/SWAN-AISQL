# This file is included by DuckDB's build system. It specifies which extension to load

# Extension from this repo (LOAD_TESTS registers test/sql/** with the unittest binary)
duckdb_extension_load(aisql
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)

# Extra extensions the SemBench / agent_bench harnesses rely on (parquet + JSON I/O, TPC-H dbgen for
# the PLOP queries). The sqllogictests need none of them.
duckdb_extension_load(parquet)
duckdb_extension_load(json)
duckdb_extension_load(tpch)
