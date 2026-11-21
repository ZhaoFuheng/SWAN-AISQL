#!/bin/bash
# The SWAN + Jev column: SWAN with ai_filter -> TypeSafe Noul, ai_classify -> TypeSafe Choice and bounded integer
# ai_score -> TypeSafe Score (jev-latest; only MOVIE q9/q10 have such a score); everything else (ai_complete /
# ai_agg, other ai_score forms, image-bearing filters and classifies) stays on gpt-5.6-luna.
# Serial, through the :4001 proxy, which answers a recorded System One call like any other recorded call, so the
# column replays from the published cache without a key. A call the proxy has not recorded goes to TypeSafe and
# needs TYPESAFE_API_KEY in ../.env.
set -u
cd "$(dirname "$0")"
export AI_MODEL=gpt-5.6-luna AI_MAX_CONCURRENCY=20 SWAN_TAG=_typesafe
key="$(grep '^TYPESAFE_API_KEY=' ../.env 2>/dev/null | cut -d= -f2-)"
export TYPESAFE_API_KEY="${key:-replay-only}"
PY="${PYTHON:-python3}"
echo "START $(date)"; curl -s http://127.0.0.1:4001/cache/stats | cut -c1-120; echo
JEV="SET ai_typesafe='filter,classify';"                 # the runners put SWAN_EXTRA_SET in front of each query
JEV_MOVIE="SET ai_typesafe='filter,classify,score';"
echo "### MOVIE ###"; ( cd MOVIE && SWAN_EXTRA_SET="$JEV_MOVIE" "$PY" swan_movie.py --serial 2>&1 | tail -14 )
echo "### ECOMM ###"; ( cd ECOMM && SWAN_EXTRA_SET="$JEV" "$PY" swan_ecomm.py --serial 2>&1 | tail -18 )
echo "### MMQA ###";  ( cd MMQA  && SWAN_EXTRA_SET="$JEV" "$PY" swan_mmqa.py  2>&1 | tail -16 )
echo "END $(date)"; curl -s http://127.0.0.1:4001/cache/stats | cut -c1-120; echo
echo "### ALL DONE ###"
