#!/bin/bash
# After the SWAN pair finishes: LOTUS ECOMM replay with true request counting (hits+misses).
while kill -0 64791 2>/dev/null; do sleep 60; done
PY=/Users/owner/opt/anaconda3/envs/swan-ai-sql/bin/python
cd "$(dirname "$0")/ECOMM" && PYTHONHASHSEED=0 $PY lotus_ecomm.py > lotus_ecomm_luna_recount.log 2>&1
echo "STAGE lotus-ecomm-recount done"
