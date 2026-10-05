# Running PLOP (Morrila) as a baseline

PLOP is the authors' DuckDB fork ("Morrila"); it is not publicly released, so it is not in this repository.
With their source tree, three small edits make it run through the cache proxy with the benchmark model, and
the runners here (`AGENTBENCH/plop_agentbench.py`, `MOVIE/plop_movie.py`) drive the resulting shell (`PLOP_BIN`, default `../../MorrilaPLOP/PLOP/build/release/duckdb`).

## Edits to the fork (`src/common/llm_query.cpp`, `src/include/duckdb/common/llm_query.hpp`)

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

Build: `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -S . -B build/release && cmake --build build/release --target shell`.

## Protocol notes

- DP cost-model mode (`DUCKDB_SEMANTIC_MODE=costmodel`), `NUM_LLM_WORKERS` = the benchmark concurrency.
- Calls are counted from the fork's own log (`llm_calls_<model>.log`, one `PROMPT:` line per request): the
  proxy's counters would include other systems running at the same time.
- Cost is estimated from the logged token counts at the gpt-5.6 list price (the fork's `/v1/responses`
  requests carry no provider cost header through the proxy).
- PLOP does not support images, so SemBench ECOMM and MMQA are not run for it; on the hybrid bench its
  prompts are the originals, so SWAN replays its verdicts through the proxy's shared-verdict alias.
