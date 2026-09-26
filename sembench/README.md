# SemBench / agent_bench harnesses

Recorded numbers to cite live in the memory index and in DESIGN.md §5; this file is the operating
protocol. Rules, in order of how often they were broken:

1. **Never rebuild the engine while a run is in flight** -- each query spawns a fresh CLI, so a
   mid-run rebuild silently mixes two binaries.
2. **Back-to-back recording for any latency claim.** Record both systems per query, LOTUS then
   SWAN, into a fresh store through a second proxy (`CACHE_PROXY_PORT=4011 CACHE_DB=<new file>`,
   alias OFF, the conda `swan-ai-sql` python -- `/usr/local/bin/python3` has no duckdb), then merge
   with `_pre_rerecord_20260923/merge_cache.py` (migrates `recorded_at`, upserts, self-verifies).
   The proxy inserts `ON CONFLICT DO NOTHING`, so the main cache cannot be "repopulated" in place.
3. **Back up the results files first.** Every runner overwrites its `*_results.json` per query.
4. **Check `pmset -g log` for a laptop sleep** across the run; find recordings that overlap it with
   `recorded_at - latency < wake AND recorded_at > sleep` (`recorded_at` is UTC) and rerun that query.
5. **A provider error aborts a LOTUS query outright** (one rejected image cost MMQA q7 18 minutes);
   rerun it against the partial store -- completed calls replay, only the remainder is fresh.
6. **Always score a run.** A call-count drop once hid a 0.929 macro-F1 with three broken queries.

## What is on disk vs. what you need to fetch

Committed: every query (SWAN, PLOP, PLOP ground-truth, LOTUS), the harnesses, gold/ground truth, the small
suite databases (`ECOMM/ecomm.db`, `MOVIE/movie.db`, `MMQA/mmqa.db`), the agent_bench datasets
(`AGENTBENCH/dataset/`, TPC-H sf0.005 parquet + book_review/yelp/googlelocal/stockindex), the per-run
summaries (`*_results.json`, comparison `.md`) and the result rows of the cited runs
(`AGENTBENCH/results/{plop,plop_gt,plop_none,lotus,swan_leaf2}`; the 52MB Q19 outputs are gzipped and
the scorer reads `.csv.gz`).

Not committed (images, raw sources, caches):

- `ECOMM/data/sf_500/images/` -- 500 product images. `ecomm_prepare.py` rebuilds `data/sf_500/` from the
  SemBench ECOMM source (Kaggle fashion-product-images); the parquet tables + `image_files.txt` are committed,
  only the image files are missing, so text-only queries (q1, q3, q5, q7) run as-is.
- `MMQA/files/mmqa/source_data/` and the image files under `files/mmqa/data/` -- from the SemBench MMQA
  release; the sf_200 CSVs that `setup.sql` reads are committed.
- `serve/.llm_cache*.duckdb` -- the recorded LLM answers (latency + cost replay). Without it every run is
  fresh against the provider; with it, reruns are free and latency-faithful.

Runners default to `../../build/release/duckdb`; set `DUCKDB_BIN` to use another build (e.g. reldebug).

Runners: `MOVIE/swan_movie.py --serial`, `ECOMM/swan_ecomm.py --serial`, `MMQA/swan_mmqa.py`;
`*/lotus_*.py`; `AGENTBENCH/swan_agentbench.py` + `eval_agentbench.py` (deterministic GT). Each takes
query ids as arguments. The 2026-09-23 re-record's driver, log, notes and pre-record backups are in
`_pre_rerecord_20260923/`.
