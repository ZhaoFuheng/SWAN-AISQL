#!/bin/bash
# Chained after the staged e2e: SWAN MOVIE with the adaptive graph scheduler, serial, no overlap.
while kill -0 92953 2>/dev/null; do sleep 60; done
echo "STAGE e2e chain finished; starting adaptive run"
PY=/Users/owner/opt/anaconda3/envs/swan-ai-sql/bin/python
AI_GRAPH_EVAL=lazy-adaptive $PY swan_movie.py --serial > swan_movie_luna_adaptive.log 2>&1
echo "STAGE adaptive finished"
echo "MOVIE ADAPTIVE DONE"
