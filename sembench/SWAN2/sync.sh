#!/bin/sh
# Refresh this copy of the SWAN 2.0 queries from a checkout of github.com/ZhaoFuheng/SWAN.
#   sembench/SWAN2/sync.sh [path-to-SWAN-checkout]   (default: ../../../SWAN_bench)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${1:-$HERE/../../../SWAN_bench}
[ -d "$SRC/queries/aisql" ] || { echo "no queries/aisql under $SRC" >&2; exit 1; }
rm -rf "$HERE/aisql" "$HERE/oracle" "$HERE/questions"
cp -R "$SRC/queries/aisql" "$SRC/queries/oracle" "$HERE/"
cp -R "$SRC/data/questions" "$HERE/questions"
echo "synced from $SRC at $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo '?'): $(find "$HERE/aisql" -name '*.sql' | wc -l | tr -d ' ') queries"
