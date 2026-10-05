"""Run one Palimpzest query program and print the result as JSON. Runs in Palimpzest's own interpreter
(setup_palimpzest.sh), so it imports nothing from this repository.

    python palimpzest_exec.py MODULE_PATH FUNCTION DATA_DIR MODEL PROXY_URL DOP [--policy MaxQuality|MinCost]
                              [--optimizer pareto|none]

MODULE_PATH is a Python file holding the query program: either SemBench's style, a `run(pz_config, data_dir)`
function (FUNCTION = "run"), or a module of functions `q1(pz_config, data_dir)`, ... (FUNCTION = the name).
The configuration is SemBench's for Palimpzest (parallel execution, 20 workers, MaxQuality) with the
Abacus's pareto plan search (`optimizer_strategy="pareto"`; SemBench ran it off; no validator, so no
sampling) and one available model, the
benchmark model through the cache proxy (`Model(model_id, api_base=...)`, which Palimpzest treats as a
self-hosted model priced at zero, so calls and cost are measured outside, at the proxy). stdout carries one
JSON object: {"columns", "rows", "stats": <numeric execution statistics>, "seconds"}; Palimpzest's own
output goes to stderr.
"""
import dataclasses
import importlib.util
import json
import math
import sys
import time


def _py(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is not None and type(value).__name__ in ("Timestamp", "NaTType", "Timedelta", "date", "datetime"):
        return None if str(value) == "NaT" else str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, default=str)
    return value


def main() -> None:
    module_path, function, data_dir, model, proxy, dop = sys.argv[1:7]
    flags = sys.argv[7:]
    policy_name = flags[flags.index("--policy") + 1] if "--policy" in flags else "MaxQuality"
    optimizer = flags[flags.index("--optimizer") + 1] if "--optimizer" in flags else "pareto"
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    import palimpzest as pz
    from palimpzest.constants import Model

    lm_model = model if "/" in model else f"openai/{model}"
    m = Model(lm_model, api_base=proxy.rstrip("/") + "/v1")
    policy = pz.MaxQuality() if policy_name == "MaxQuality" else pz.MinCost()
    config = pz.QueryProcessorConfig(policy=policy, optimizer_strategy=optimizer, execution_strategy="parallel",
                                     max_workers=int(dop), join_parallelism=int(dop), available_models=[m],
                                     progress=False, verbose=False)

    spec = importlib.util.spec_from_file_location("pz_query_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, function)

    t0 = time.time()
    output = fn(config, data_dir)
    seconds = round(time.time() - t0, 2)
    if isinstance(output, dict):  # SemBench's post-processed form: {"results": DataFrame, "execution_stats": ...}
        df, stats = output["results"], output["execution_stats"]
    else:
        df, stats = output.to_df(), output.execution_stats
    numeric = {k: v for k, v in (dataclasses.asdict(stats).items() if dataclasses.is_dataclass(stats) else vars(stats).items())
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out = {"columns": [str(c) for c in df.columns],
           "rows": [[_py(v) for v in row] for row in df.itertuples(index=False, name=None)],
           "stats": numeric, "seconds": seconds}
    json.dump(out, real_stdout, default=str)
    real_stdout.write("\n")


if __name__ == "__main__":
    main()
