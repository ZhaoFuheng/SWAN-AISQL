#!/bin/bash
set -u
SEM=/Users/owner/Desktop/research/swan-ai-sql/swan-ai-sql_v2/extension/aisql/sembench
SERVE=/Users/owner/Desktop/research/swan-ai-sql/swan-ai-sql_v2/extension/aisql/serve
PY=/Users/owner/opt/anaconda3/envs/swan-ai-sql/bin/python
L=$SEM/_pre_rerecord_20260923/rerecord.log
export AI_PROXY_URL=http://127.0.0.1:4011 AI_MODEL=gpt-5.6-luna AI_MAX_CONCURRENCY=20
cd $SERVE
CACHE_PROXY_PORT=4011 CACHE_DB=$SERVE/.llm_cache_rerecord_20260923.duckdb CACHE_UPSTREAM=http://127.0.0.1:4000 \
  CACHE_EMBED_UPSTREAM=http://127.0.0.1:4002 nohup $PY ai_cache_server.py > $SEM/_pre_rerecord_20260923/proxy_reruns.log 2>&1 &
PROXY=$!; sleep 3
curl -sf http://127.0.0.1:4011/cache/stats >/dev/null || { echo "proxy failed to start" | tee -a $L; exit 1; }
echo "=== RERUNS (partial-cache / replay) ===" | tee -a $L
rerun() { # suite query note
  local low=$(echo $1 | tr A-Z a-z) t0=$(date +%s); cd $SEM/$1
  local out=$($PY lotus_${low}.py $2 2>&1 | tr '\r' '\n')
  local line=$(echo "$out" | grep -E "^ *$2 +lat=" | tail -1)
  [ -z "$line" ] && line="NO RESULT LINE: $(echo "$out" | grep -iE "error|traceback" | tail -2 | tr '\n' ' ')"
  printf "%s %-5s LOTUS %-4s wall=%4ss | %s   [%s]\n" "$(date +%H:%M:%S)" $1 $2 $(($(date +%s)-t0)) "$line" "$3" | tee -a $L
}
rerun MMQA q7 "rerun after provider image rejection; 7,999 calls replayed from this session"
rerun MOVIE q5 "replay rerun: original wall time included a 71s laptop sleep"
echo "=== rerun proxy stats ===" | tee -a $L; curl -s http://127.0.0.1:4011/cache/stats | tee -a $L; echo | tee -a $L
kill $PROXY; sleep 2
echo "### RERUNS DONE ###" | tee -a $L
