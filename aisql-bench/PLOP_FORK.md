# Running PLOP (Morrila) as a baseline

PLOP is the authors' DuckDB fork, "Morrila", public at https://github.com/DMoRanSky/Morrila (MIT). The
runners here (`AGENTBENCH/plop_agentbench.py`, `MOVIE/plop_movie.py`) and SWAN_bench's `plop` adapter drive its
shell; the default location is a clone next to this repository, `../Morrila`, built to `build/release/duckdb`
(`PLOP_ROOT` / `PLOP_BIN` / `--plop-bin` override it).

## Setup

```sh
cd <parent of SWAN-AISQL>
git clone https://github.com/DMoRanSky/Morrila          # measured at commit 56780d9 (2026-10-13)
cd Morrila
git apply ../SWAN-AISQL/aisql-bench/plop_fork.patch      # the three edits below
rm -rf build/release                                      # the repo ships a CMake cache from another machine
echo 'duckdb_extension_load(tpch)' > extension/extension_config_local.cmake   # Q15-Q19, Q25-Q30 LOAD tpch
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -S . -B build/release
cmake --build build/release --target shell tpch_loadable_extension
```

(The Makefile's `CORE_EXTENSIONS=` path needs `.github/config/*.cmake`, which the public repo does not ship; the
local extension config is the supported alternative.)

## The three edits (`plop_fork.patch`: `src/common/llm_query.cpp`, `src/include/duckdb/common/llm_query.hpp`)

1. **Endpoint from the environment** — in `LLMCallManager::LLMCallManager`:

   ```cpp
   llm_api_url_ = std::getenv("LLM_API_URL") ? std::getenv("LLM_API_URL") : "https://api.openai.com/v1/responses";
   ```

   The runners set `LLM_API_URL=<proxy>/v1/responses`; the proxy forwards the Responses API to litellm.

2. **Model from the environment** — in `getLLMCallManagerInstance(const std::string &requested_model)`:

   ```cpp
   const std::string model_name = std::getenv("LLM_MODEL") ? std::string(std::getenv("LLM_MODEL")) : requested_model;
   ```

   so every semantic function uses the benchmark model (`gpt-5.6-luna`) instead of the fork's default.

3. **Answer text wherever the message is** — the fork reads the answer at `output[1].content[0].text`, which
   assumes a reasoning item precedes the message. The message is at `output[0]` when the model emits no
   reasoning item (gpt-5.6-luna through litellm), and every verdict parsed as false. In `LLMResponse`:

   ```cpp
   auto extract_answer_text_() -> std::string {
       std::string t = json_extract_string_from_keys_(text, {"output", "1", "content", "0", "text"});
       if (t.empty()) {
           t = json_extract_string_from_keys_(text, {"output", "0", "content", "0", "text"});
       }
       return t;
   }
   ```

   and the four `parse_to_*_()` methods call it in place of the fixed-index extraction.

None of the edits touches the optimizer (`src/optimizer/semantic.cpp`) or the SQL surface.

## Protocol notes

- DP cost-model mode (`DUCKDB_SEMANTIC_MODE=costmodel`), `NUM_LLM_WORKERS` = the benchmark concurrency.
- Calls are counted from the fork's own log (`llm_calls_<model>.log`, one `PROMPT:` line per request): the
  proxy's counters would include other systems running at the same time.
- Cost is estimated from the logged token counts at the gpt-5.6 list price (the fork's `/v1/responses`
  requests carry no provider cost header through the proxy).
- PLOP does not support images, so SemBench ECOMM and MMQA are not run for it; on the hybrid bench its
  prompts are the originals, so SWAN replays its verdicts through the proxy's shared-verdict alias.
- The hybrid bench's 30 queries and data ship in the fork under `test/semantic/agent_bench/`.
- Q29 and Q30 have LIMITs inside CTEs with no ORDER BY, so the rows they ask about depended on the engine (the
  public build draws different nations on Q29 than the pre-release tree did). Every system's copy orders those
  LIMITs by key (`plop_queries/`, `plop_gt_queries/`, `swan_queries/`, `blendsql_queries/`, `thalamusdb_queries/`,
  `lotus_queries/`); PLOP runs from `plop_queries/` (`PLOP_QUERY_DIR`) for those two, and the ground truth
  (`results/plop_gt` = public build, `DUCKDB_SEMANTIC_MODE=none`, `plop_gt_queries/`) is generated the same way.
