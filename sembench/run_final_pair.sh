#!/bin/bash
# Default SWAN (adaptive streaming + graph LIMIT + existential + hedging + global gate,
# query-scoped cache) on both suites, serial, no overlap -- the LOTUS-vs-default-SWAN numbers.
PY=/Users/owner/opt/anaconda3/envs/swan-ai-sql/bin/python
cd "$(dirname "$0")/ECOMM" && $PY swan_ecomm.py --serial > swan_ecomm_luna_final.log 2>&1
echo "STAGE ecomm done"
cd ../MOVIE && $PY swan_movie.py --serial > swan_movie_luna_final.log 2>&1
echo "STAGE movie done"
echo "FINAL PAIR DONE"
