#!/bin/bash
# Run the MLP-feature A/B (factored [pred,input,cos] vs single emb(prompt)) on the 9 selected
# govreport queries WITH latency replay on, so the online training happens at a faithful cadence.
set -uo pipefail
cd /Users/owner/Desktop/research/swan-ai-sql/swan-ai-sql
PY=/Users/owner/opt/anaconda3/envs/swan-ai-sql/bin/python
export BENCH_TABLE=govreport

echo "=== FACTORED [pred,input,cos] (replay) ==="
AI_MLP_FEATURE=factored "$PY" govreport/bench.py > govreport/bench_featA_replay.out 2> govreport/bench_featA_replay.err
cp govreport/bench_results.json govreport/bench_results_featA_replay.json

echo "=== PROMPT emb(full prompt) (replay) ==="
AI_MLP_FEATURE=prompt "$PY" govreport/bench.py > govreport/bench_featB_replay.out 2> govreport/bench_featB_replay.err
cp govreport/bench_results.json govreport/bench_results_featB_replay.json

echo "=== DONE ==="
echo "FACTORED: $(grep Summary govreport/bench_featA_replay.out)"
echo "PROMPT:   $(grep Summary govreport/bench_featB_replay.out)"
