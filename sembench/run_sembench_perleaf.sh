#!/bin/bash
# SWAN per-leaf build (per-leaf factorization + overlap 8, defaults) on all three suites,
# serial, replayed through the :4001 alias proxy (recorded latency/cost from the 2026-09-23 re-record).
set -u
cd "$(dirname "$0")"
export AI_MODEL=gpt-5.6-luna AI_MAX_CONCURRENCY=20
echo "START $(date)"; curl -s http://127.0.0.1:4001/cache/stats | cut -c1-120; echo
echo "### MMQA ###";  ( cd MMQA  && /usr/local/bin/python3 swan_mmqa.py  2>&1 | tail -16 )
echo "### ECOMM ###"; ( cd ECOMM && /usr/local/bin/python3 swan_ecomm.py --serial 2>&1 | tail -18 )
echo "### MOVIE ###"; ( cd MOVIE && /usr/local/bin/python3 swan_movie.py --serial 2>&1 | tail -14 )
echo "END $(date)"; curl -s http://127.0.0.1:4001/cache/stats | cut -c1-120; echo
echo "### ALL DONE ###"
