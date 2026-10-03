#!/bin/bash
# Start the serving stack the engine talks to, reading provider keys from the repo's .env:
#
#   engine --(SET ai_endpoint=:4001)--> ai_cache_server.py --> litellm (:4000) --> OpenAI (gpt-5.6-luna)
#                                            '--> https://api.typesafe.ai (Jev; only with SET ai_typesafe)
#   engine --(ai_embed_endpoint=:4002)--> ai_embed_server.py (selectivity features; optional but recommended)
#
# Usage:  serve/start_stack.sh [--no-embed]        (from anywhere; logs go to serve/*.log)
# Stop:   pkill -f ai_cache_server.py; pkill -f 'litellm --config'; pkill -f ai_embed_server.py
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
if [ -f "$ROOT/.env" ]; then
	set -a; . "$ROOT/.env"; set +a
fi
PY="${PYTHON:-python3}"

# Without a provider key the stack is replay-only: the cache proxy answers recorded requests (see
# serve/fetch_cache.sh) and a request it has not seen fails with an upstream error instead of costing money.
if [ -z "${OPENAI_API_KEY:-}" ]; then
	echo "no OPENAI_API_KEY in $ROOT/.env: replay-only (recorded requests answer from serve/.llm_cache.duckdb, new ones fail)"
elif ! curl -sf -m 2 http://127.0.0.1:4000/health >/dev/null 2>&1; then
	nohup litellm --config "$HERE/litellm.config.yaml" --port 4000 > "$HERE/litellm.log" 2>&1 &
	echo "litellm      :4000  (pid $!, log serve/litellm.log)"
else
	echo "litellm      :4000  already running"
fi
if ! curl -sf -m 2 http://127.0.0.1:4001/cache/stats >/dev/null 2>&1; then
	# CACHE_ALIAS_CHAT=1 is the agent_bench shared-verdict mode (SWAN replays PLOP's recorded samples keyed on
	# the user text alone). Off by default: for ordinary use every request must key on its full body.
	( cd "$HERE" && nohup "$PY" ai_cache_server.py > "$HERE/cache_proxy.log" 2>&1 & echo "cache proxy  :4001  (pid $!, log serve/cache_proxy.log, alias=${CACHE_ALIAS_CHAT:-0})" )
else
	echo "cache proxy  :4001  already running"
fi
if [ "${1:-}" != "--no-embed" ]; then
	if ! curl -sf -m 2 http://127.0.0.1:4002/ >/dev/null 2>&1; then
		( cd "$HERE" && nohup "$PY" ai_embed_server.py > "$HERE/embed.log" 2>&1 & echo "embed server :4002  (pid $!, log serve/embed.log)" )
	else
		echo "embed server :4002  already running"
	fi
fi
sleep 3
curl -sf -m 5 http://127.0.0.1:4001/cache/stats | cut -c1-100 && echo || echo "cache proxy did not come up -- see serve/cache_proxy.log (needs: pip install duckdb)"
