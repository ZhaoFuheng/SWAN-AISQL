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
- **Input-selecting LIMITs are ordered by key.** A LIMIT with no ORDER BY that sits *below* a semantic function
  decides which rows the function is asked about, so the input depended on the engine and the machine (the
  public build draws different nations on Q29 than the pre-release tree did; the arm64 rebuild of 2026-11-09
  drew different customers for BlendSQL on Q15 and Q19). Every such LIMIT is ordered by its key in every
  system's copy (`plop_queries/`, `plop_gt_queries/`, `swan_queries/`, `blendsql_queries/`,
  `lotus_queries/`), so all systems are evaluated on the same input:

  | queries | LIMIT | ordered by | since |
  |---|---|---|---|
  | Q29, Q30 | the CTE key LIMITs | the key (`ps_suppkey`, `s_suppkey, s_nationkey`, `s_nationkey`, `l_partkey`, `l_suppkey`) | 2026-10-15 |
  | Q15, Q16, Q17, Q19, Q25 | `customer_sample`: `SELECT * FROM customer LIMIT k` | `c_custkey` | 2026-11-10 |
  | Q18 | `nation_sample`: `SELECT * FROM nation LIMIT 25` | `n_nationkey` | 2026-11-10 |
  | Q27 | `SELECT DISTINCT ps_suppkey FROM joined LIMIT 80` | `ps_suppkey` | 2026-11-10 |
  | Q28 | `SELECT DISTINCT o_custkey FROM joined LIMIT 60` | `o_custkey` | 2026-11-10 |

  A LIMIT *above* a semantic function (the trailing `LIMIT k` of Q1-Q3) is left unordered: it is what lets a
  system stop early, and those queries are scored on soundness and completeness, not row identity.
  PLOP must run these ten queries from `plop_queries/` (`PLOP_QUERY_DIR=<abs path>/AGENTBENCH/plop_queries`):
  the fork's own `test/semantic/agent_bench/` copies are unordered. The ground truth (`results/plop_gt` =
  public build, `DUCKDB_SEMANTIC_MODE=none`, `plop_gt_queries/`) is generated the same way.
  The 2026-11-10 ordering changes the input of Q15-Q19, Q25, Q27 and Q28, so their ground truth and every
  system's rows for those eight queries were re-recorded on 2026-11-11; the published hybrid-bench table
  is measured on the ordered queries.
