#!/bin/bash
# Regression pass on the extension-template build: all three SemBench suites, serial, replayed through the
# :4001 proxy, results tagged _port (compare against the untagged per-leaf run with compare_tags.py "" _port).
set -u
cd "$(dirname "$0")"
export DUCKDB_BIN="$PWD/../build/reldebug/duckdb"
export AI_MODEL=gpt-5.6-luna AI_MAX_CONCURRENCY=20 SWAN_TAG=_port
echo "START $(date)  bin=$DUCKDB_BIN"; curl -s http://127.0.0.1:4001/cache/stats | cut -c1-120; echo
echo "### MMQA ###";  ( cd MMQA  && ${PYTHON:-/usr/local/bin/python3} swan_mmqa.py  2>&1 | tail -16 )
echo "### ECOMM ###"; ( cd ECOMM && ${PYTHON:-/usr/local/bin/python3} swan_ecomm.py --serial 2>&1 | tail -18 )
echo "### MOVIE ###"; ( cd MOVIE && ${PYTHON:-/usr/local/bin/python3} swan_movie.py --serial 2>&1 | tail -14 )
echo "END $(date)"; curl -s http://127.0.0.1:4001/cache/stats | cut -c1-120; echo
echo "### ALL DONE ###"
