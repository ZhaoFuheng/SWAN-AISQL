#!/bin/bash
# Download the published replay cache so every benchmark in this repository (SemBench MOVIE / ECOMM / MMQA,
# the agent_bench hybrid queries, SWAN 2.0) replays its recorded answers, latency and cost without a
# provider key. The cache proxy picks the file up at serve/.llm_cache.duckdb.
#
# Usage:  serve/fetch_cache.sh [URL]
#
# Without a URL the script asks Zenodo for the LATEST version of the record (concept DOI
# 10.5281/zenodo.23112764; published versions are immutable, so an updated cache is always a new version),
# checks the download against the md5 Zenodo publishes for it, and unpacks it (the published file is the
# gzip-compressed DuckDB file: about 1.1 GB to download, about 5 GB on disk). A download cut short resumes
# where it stopped, also when the script is run again. A URL ending in .gz is unpacked the same way; any
# other URL is taken as the DuckDB file itself.
#
# Replay serves byte-identical requests only: the repository's queries, model gpt-5.6-luna, the engine's
# default prompts and schemas. An edited query sends a fresh request, which needs OPENAI_API_KEY in .env.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/.llm_cache.duckdb"
LATEST_API="https://zenodo.org/api/records/23112765/versions/latest"   # any version's id resolves to the latest
FILE="swan_replay_cache.duckdb.gz"   # the name in the latest version (swan_replay_cache[_YYYYMMDD].duckdb[.gz]) is read from the record below

if [ -s "$DEST" ]; then
	echo "$DEST exists ($(du -h "$DEST" | cut -f1)); move it away first if you want the published cache instead."
	exit 1
fi
if curl -sf -m 2 http://127.0.0.1:4001/cache/stats >/dev/null 2>&1; then
	echo "a cache proxy is running on :4001; stop it first (pkill -f ai_cache_server.py)"
	exit 1
fi

URL="${1:-}"
MD5=""
if [ -z "$URL" ]; then
	record=$(curl -sfL -m 30 "$LATEST_API") || { echo "could not reach Zenodo ($LATEST_API)"; exit 1; }
	id=$(printf '%s' "$record" | grep -o 'https://zenodo.org/records/[0-9]*' | head -1 | grep -o '[0-9]*$')
	key=$(printf '%s' "$record" | grep -oE '"key": ?"swan_replay_cache(_[0-9]{8})?\.duckdb(\.gz)?"' | head -1 | grep -oE 'swan_replay_cache(_[0-9]{8})?\.duckdb(\.gz)?')
	[ -n "$key" ] && FILE="$key"
	MD5=$(printf '%s' "$record" | grep -o "\"$FILE\"[^}]*\"md5:[0-9a-f]*\"" | grep -o '"md5:[0-9a-f]*"' | head -1 | tr -d '"' | cut -d: -f2)
	[ -n "$MD5" ] || MD5=$(printf '%s' "$record" | grep -o '"md5:[0-9a-f]*"' | head -1 | tr -d '"' | cut -d: -f2)
	[ -n "$id" ] || { echo "could not find the latest record id in Zenodo's answer"; exit 1; }
	URL="https://zenodo.org/records/$id/files/$FILE?download=1"
	echo "latest version: https://zenodo.org/records/$id"
else
	FILE="$(basename "${URL%%\?*}")"
fi
# One partial file per file name, so a cut download is resumed (curl -C -) and never resumed into from another
# version. A transfer slower than 1 KB/s for a minute counts as cut. After five attempts the script stops; run it
# again and it resumes from the same place.
PART="$DEST.download-$FILE"
echo "downloading $URL"
n=1
until curl -L --fail --progress-bar -C - --speed-limit 1024 --speed-time 60 -o "$PART" "$URL"; do
	rc=$?
	[ $rc -eq 33 ] && rm -f "$PART"   # the server cannot resume: the next attempt starts over
	[ $n -lt 5 ] || { echo "download stopped $n times (curl exit $rc); run the script again to resume"; exit 1; }
	echo "download stopped (curl exit $rc); attempt $((n + 1)) of 5 in 10 s"
	n=$((n + 1))
	sleep 10
done
if [ -n "$MD5" ]; then
	if command -v md5sum >/dev/null 2>&1; then sum=$(md5sum "$PART" | cut -d' ' -f1)
	elif command -v md5 >/dev/null 2>&1; then sum=$(md5 -q "$PART")
	else sum="$MD5"; fi
	[ "$sum" = "$MD5" ] || { echo "checksum mismatch ($sum, Zenodo says $MD5); download again"; rm -f "$PART"; exit 1; }
fi
case "$URL" in
	*.gz|*.gz\?*) echo "unpacking"; gzip -dc "$PART" > "$DEST.part" && rm -f "$PART" && mv "$DEST.part" "$DEST" ;;
	*) mv "$PART" "$DEST" ;;
esac
echo "replay cache: $DEST ($(du -h "$DEST" | cut -f1))"
echo "start the stack with serve/start_stack.sh; every recorded request now replays at \$0"
