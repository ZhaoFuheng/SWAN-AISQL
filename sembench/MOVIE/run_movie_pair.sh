#!/bin/bash
PY=/Users/owner/opt/anaconda3/envs/swan-ai-sql/bin/python
AI_GRAPH_EVAL=staged $PY swan_movie.py --serial > swan_movie_luna_staged.log 2>&1
echo "STAGE staged done"
$PY swan_movie.py --serial > swan_movie_luna_adaptive.log 2>&1
echo "STAGE adaptive done"
echo "MOVIE PAIR DONE"
