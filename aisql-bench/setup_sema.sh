#!/bin/bash
# Sema (VLDB 2026, github.com/BITQiKangK/Sema: a DuckDB fork with s'...' semantic expressions) built the way its
# runners use it: ../../Sema/build/release/duckdb next to this checkout (override the location with SEMA_DIR).
#   - pinned to commit a86cb62 ("init sema 1.0 version", 2025-12-25);
#   - sema_fork.patch: on non-Linux systems the LLM client polls the sockets libcurl registered instead of using
#     epoll (Linux-only); Linux keeps the original code path. Two environment variables set the client's limits,
#     unchanged when unset: SEMA_LLM_MAX_CONNECTIONS (requests in flight, 50 per host) and SEMA_LLM_TIMEOUT_S
#     (request timeout, 30 s; libcurl counts it from when a request is queued);
#   - nlohmann/json 3.11.3 (single header, checksum below), which Sema expects as a system header;
#   - the vss, parquet and httpfs extensions are skipped (they do not compile/link with a current Apple
#     toolchain, and no benchmark query uses them; Sema's model calls go through libcurl, not httpfs).
# Needs git, cmake, ninja, curl and a C++ toolchain. Takes ~20 minutes.
set -eu
cd "$(dirname "$0")"
HERE="$(pwd)"
SEMA_DIR="${SEMA_DIR:-$HERE/../../Sema}"
COMMIT=a86cb62297048fc92afbfb173d4387174d6d53a4
JSON_URL=https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
JSON_SHA256=9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6

if [ ! -d "$SEMA_DIR/.git" ]; then
  mkdir -p "$SEMA_DIR"
  git -C "$SEMA_DIR" init -q
  git -C "$SEMA_DIR" remote add origin https://github.com/BITQiKangK/Sema
  git -C "$SEMA_DIR" fetch -q --depth 1 origin "$COMMIT"
  git -C "$SEMA_DIR" checkout -q FETCH_HEAD
fi
[ "$(git -C "$SEMA_DIR" rev-parse HEAD)" = "$COMMIT" ] || { echo "$SEMA_DIR is not at $COMMIT"; exit 1; }
if git -C "$SEMA_DIR" apply --check "$HERE/sema_fork.patch" 2>/dev/null; then
  git -C "$SEMA_DIR" apply "$HERE/sema_fork.patch"
elif ! git -C "$SEMA_DIR" apply --check -R "$HERE/sema_fork.patch" 2>/dev/null; then
  echo "sema_fork.patch neither applies nor is already applied in $SEMA_DIR"; exit 1
fi

DEPS="$SEMA_DIR/build/deps/include"
mkdir -p "$DEPS/nlohmann"
if [ ! -f "$DEPS/nlohmann/json.hpp" ]; then
  curl -sSL --fail -o "$DEPS/nlohmann/json.hpp" "$JSON_URL"
fi
echo "$JSON_SHA256  $DEPS/nlohmann/json.hpp" | shasum -a 256 -c -

cmake -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS=-I$DEPS" "-DSKIP_EXTENSIONS=vss;parquet;httpfs" \
  -S "$SEMA_DIR" -B "$SEMA_DIR/build/release"
cmake --build "$SEMA_DIR/build/release" --target shell
echo "Sema ready: $SEMA_DIR/build/release/duckdb"
