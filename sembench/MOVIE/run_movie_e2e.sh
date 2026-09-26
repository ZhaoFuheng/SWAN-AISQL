#!/bin/bash
# End-to-end MOVIE comparison: wait for the in-flight LOTUS replay, then SWAN serial -- never
# overlapping, so both engines' latency-faithful replays are clean.
while kill -0 91580 2>/dev/null; do sleep 30; done
echo "STAGE lotus finished"
PY=/Users/owner/opt/anaconda3/envs/swan-ai-sql/bin/python
$PY swan_movie.py --serial > swan_movie_luna_recount.log 2>&1
echo "STAGE swan finished"
echo "MOVIE E2E DONE"
