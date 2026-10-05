#!/bin/bash
# SWAN with ai_filter -> TypeSafe Noul, ai_classify -> TypeSafe Choice and bounded integer ai_score ->
# TypeSafe Score (jev-latest); everything else (ai_complete/ai_agg, other ai_score forms, image-bearing
# filters/classifies) on gpt-5.6-luna replayed from the
# 2026-09-23 re-record. Serial, through the :4001 proxy (records the fresh System One calls).
set -u
cd "$(dirname "$0")"
export AI_MODEL=gpt-5.6-luna AI_MAX_CONCURRENCY=20 AI_TYPESAFE='filter,classify,score' SWAN_TAG=_typesafe
export TYPESAFE_API_KEY="$(grep '^TYPESAFE_API_KEY=' "$(dirname "$0")/../.env" | cut -d= -f2-)"
echo "START $(date)"; curl -s http://127.0.0.1:4001/cache/stats | cut -c1-120; echo
echo "### MOVIE ###"; ( cd MOVIE && ${PYTHON:-/usr/local/bin/python3} swan_movie.py --serial 2>&1 | tail -14 )
echo "### ECOMM ###"; ( cd ECOMM && ${PYTHON:-/usr/local/bin/python3} swan_ecomm.py --serial 2>&1 | tail -18 )
echo "### MMQA ###";  ( cd MMQA  && ${PYTHON:-/usr/local/bin/python3} swan_mmqa.py  2>&1 | tail -16 )
echo "END $(date)"; curl -s http://127.0.0.1:4001/cache/stats | cut -c1-120; echo
echo "### ALL DONE ###"
