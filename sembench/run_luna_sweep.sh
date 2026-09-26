#!/bin/bash
# Sequential SemBench rerun on gpt-5.6-luna (reasoning_effort=low via litellm), both engines.
# Fresh calls route :4001 -> :4000 -> OpenAI and are recorded into the server cache (responses,
# latency, cost), so future runs replay for free.
set -x
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/Users/owner/opt/anaconda3/envs/swan-ai-sql/bin/python
export PYTHONHASHSEED=0
cd "$HERE/ECOMM"  && $PY swan_ecomm.py --serial  > swan_ecomm_luna.log  2>&1 && echo "STAGE ecomm-swan done"
cd "$HERE/ECOMM"  && $PY lotus_ecomm.py          > lotus_ecomm_luna.log 2>&1 && echo "STAGE ecomm-lotus done"
cd "$HERE/MOVIE"  && $PY swan_movie.py --serial  > swan_movie_luna.log  2>&1 && echo "STAGE movie-swan done"
cd "$HERE/MOVIE"  && $PY lotus_movie.py          > lotus_movie_luna.log 2>&1 && echo "STAGE movie-lotus done"
cd "$HERE/MMQA"   && $PY swan_mmqa.py            > swan_mmqa_luna.log   2>&1 && echo "STAGE mmqa-swan done"
cd "$HERE/MMQA"   && $PY lotus_mmqa.py           > lotus_mmqa_luna.log  2>&1 && echo "STAGE mmqa-lotus done"
echo "LUNA SWEEP COMPLETE"
