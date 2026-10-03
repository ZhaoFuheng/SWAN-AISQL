#!/bin/bash
# Download the published replay cache so every benchmark in this repository (SemBench MOVIE / ECOMM / MMQA,
# the agent_bench hybrid queries, SWAN 2.0) replays its recorded answers, latency and cost without a
# provider key. The cache proxy picks the file up at serve/.llm_cache.duckdb.
#
# Usage:  serve/fetch_cache.sh [URL]       (default: the Zenodo record below)
#
# Replay serves byte-identical requests only: the repository's queries, model gpt-5.6-luna, the engine's
# default prompts and schemas. An edited query sends a fresh request, which needs OPENAI_API_KEY in .env.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
URL="${1:-https://zenodo.org/records/23112765/files/swan_replay_cache.duckdb?download=1}"
DEST="$HERE/.llm_cache.duckdb"

if [ -s "$DEST" ]; then
	echo "$DEST exists ($(du -h "$DEST" | cut -f1)); move it away first if you want the published cache instead."
	exit 1
fi
if curl -sf -m 2 http://127.0.0.1:4001/cache/stats >/dev/null 2>&1; then
	echo "a cache proxy is running on :4001; stop it first (pkill -f ai_cache_server.py)"
	exit 1
fi
echo "downloading $URL"
curl -L --fail --progress-bar -o "$DEST.part" "$URL"
if [ -z "${1:-}" ] && command -v md5sum >/dev/null 2>&1; then
	sum=$(md5sum "$DEST.part" | cut -d' ' -f1)
	[ "$sum" = "1b890992d80ee90222a5b194b9da6553" ] || { echo "checksum mismatch ($sum); download again"; exit 1; }
elif [ -z "${1:-}" ] && command -v md5 >/dev/null 2>&1; then
	sum=$(md5 -q "$DEST.part")
	[ "$sum" = "1b890992d80ee90222a5b194b9da6553" ] || { echo "checksum mismatch ($sum); download again"; exit 1; }
fi
mv "$DEST.part" "$DEST"
echo "replay cache: $DEST ($(du -h "$DEST" | cut -f1))"
echo "start the stack with serve/start_stack.sh; every recorded request now replays at \$0"
