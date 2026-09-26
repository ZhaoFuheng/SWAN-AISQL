#!/bin/bash
# Back-to-back re-record of SemBench: for each query, LOTUS then SWAN, fresh, into one new cache.
set -u
SEM=/Users/owner/Desktop/research/swan-ai-sql/swan-ai-sql_v2/extension/aisql/sembench
SERVE=/Users/owner/Desktop/research/swan-ai-sql/swan-ai-sql_v2/extension/aisql/serve
PY=/Users/owner/opt/anaconda3/envs/swan-ai-sql/bin/python
DB=$SERVE/.llm_cache_rerecord_20260923.duckdb
export AI_PROXY_URL=http://127.0.0.1:4011 AI_MODEL=gpt-5.6-luna AI_MAX_CONCURRENCY=20
cd $SERVE
CACHE_PROXY_PORT=4011 CACHE_DB=$DB CACHE_UPSTREAM=http://127.0.0.1:4000 CACHE_EMBED_UPSTREAM=http://127.0.0.1:4002 \
  nohup $PY ai_cache_server.py > $SEM/_pre_rerecord_20260923/proxy.log 2>&1 &
PROXY=$!; sleep 3
curl -sf http://127.0.0.1:4011/cache/stats >/dev/null || { echo "proxy failed to start"; exit 1; }
echo "proxy pid $PROXY db $DB"
run() { # suite system query
  local suite=$1 sys=$2 q=$3 low=$(echo $1 | tr A-Z a-z) t0=$(date +%s)
  cd $SEM/$suite
  if [ $sys = LOTUS ]; then out=$($PY lotus_${low}.py $q 2>&1 | tr '\r' '\n')
  else serial=""; [ $suite != MMQA ] && serial="--serial"
       out=$(/usr/local/bin/python3 swan_${low}.py $serial $q 2>&1); fi
  local line=$(echo "$out" | grep -E "^ *$q +lat=" | tail -1)
  [ -z "$line" ] && line="NO RESULT LINE: $(echo "$out" | grep -iE "error|traceback" | tail -2 | tr '\n' ' ')"
  printf "%s %-5s %-5s %-4s wall=%4ss | %s\n" "$(date +%H:%M:%S)" $suite $sys $q $(($(date +%s)-t0)) "$line"
}
for q in q1 q2a q2b q3a q3f q4 q5 q6a q6b q6c q7; do run MMQA LOTUS $q; run MMQA SWAN $q; done
for i in $(seq 1 14); do run ECOMM LOTUS q$i; run ECOMM SWAN q$i; done
for i in $(seq 1 10); do run MOVIE LOTUS q$i; run MOVIE SWAN q$i; done
echo "=== proxy stats ==="; curl -s http://127.0.0.1:4011/cache/stats; echo
kill $PROXY; sleep 2
echo "### ALL DONE ###"
