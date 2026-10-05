"""Run one hybrid-bench program on Palimpzest and write its result. Runs in Palimpzest's own interpreter
(setup_palimpzest.sh), so it imports nothing from this repository.

    python palimpzest_hybrid_exec.py PROGRAM.py DATASET_DIR OUT.csv MODEL PROXY_URL DOP [--optimizer pareto|none] [--policy ...]

PROGRAM.py is one of the hybrid bench's LOTUS programs (AGENTBENCH/lotus_queries/QN.py: pandas for the
relational parts, `df.sem_filter("<instruction>\\nLabel: {column}\\n...")` and `df.sem_map(...)` for the
semantic ones). They run unchanged: this script gives pandas frames the same two methods backed by Palimpzest,
so the program's written order is Palimpzest's too and every semantic operator is a Palimpzest program
(sem_filter / sem_add_columns over a MemoryDataset of the frame) that its Abacus optimizer plans on each
run. The prompt's `Label: {column}` lines become the operator's `depends_on` columns, shown to the model by
Palimpzest under their names (its own form of the labelled context); what remains of the prompt is the
operator's instruction. The result frame is written with pandas, as the LOTUS runner writes its own. stdout
carries {"rows", "seconds", "operators"}; Palimpzest's own output goes to stderr.
"""
import importlib.util
import json
import os
import re
import sys
import time

CSV = {
    "books_info": "book_review/books_info.csv", "reviews": "book_review/reviews.csv",
    "decades": "book_review/decades.csv", "review_context": "book_review/review_context.csv",
    "gl_business": "googlelocal/business_description.csv", "gl_review": "googlelocal/review.csv",
    "yelp_review": "yelp/review.csv", "yelp_tip": "yelp/tip.csv", "yelp_user": "yelp/user.csv",
    "yelp_business": "yelp/yelp_business_csv/business.csv",
}
PARQUET = {t: f"tpch/{t}.parquet" for t in ["part", "supplier", "customer", "lineitem", "orders", "nation", "region", "partsupp"]}
_PLACEHOLDER = re.compile(r"\{(\w+)\}")
# a labelled field is "Label: {column}" at a line start or after a "; " / ", " separator; an instruction
# sentence that happens to end in a colon before a placeholder is not a label and stays in the instruction
_LABELLED = re.compile(r"(?m)(?:^|(?<=[;,] ))[ \t]*[A-Za-z][A-Za-z /]*:[ \t]*\{\w+\}[ \t]*[;,]?")


def split_prompt(template: str) -> tuple[str, list[str]]:
    """(instruction, depends_on columns) of a LOTUS prompt template."""
    columns = list(dict.fromkeys(_PLACEHOLDER.findall(template)))
    text = _LABELLED.sub(" ", template)
    text = _PLACEHOLDER.sub(" ", text)
    return " ".join(text.split()), columns


def main() -> None:
    program, data_dir, out_csv, model, proxy, dop = sys.argv[1:7]
    flags = sys.argv[7:]
    optimizer = flags[flags.index("--optimizer") + 1] if "--optimizer" in flags else "pareto"
    policy_name = flags[flags.index("--policy") + 1] if "--policy" in flags else "MaxQuality"
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    import palimpzest as pz
    import pandas as pd
    from palimpzest.constants import Model

    lm_model = model if "/" in model else f"openai/{model}"
    m = Model(lm_model, api_base=proxy.rstrip("/") + "/v1")
    policy = pz.MaxQuality() if policy_name == "MaxQuality" else pz.MinCost()
    state = {"n": 0, "operators": []}

    def config():
        return pz.QueryProcessorConfig(policy=policy, optimizer_strategy=optimizer, execution_strategy="parallel",
                                       max_workers=int(dop), join_parallelism=int(dop), available_models=[m],
                                       progress=False, verbose=False)

    def dataset(df, columns):
        state["n"] += 1
        vals = df[columns].copy()
        for c in columns:  # text, as the LOTUS prompt shows the value (Decimal / Timestamp are not JSON-serialisable)
            vals[c] = vals[c].map(lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v))
        vals.insert(0, "swan_i", range(len(df)))
        return pz.MemoryDataset(id=f"hybrid_{state['n']}", vals=vals.reset_index(drop=True))

    def sem_filter(self, template, **kwargs):
        instruction, columns = split_prompt(template)
        state["operators"].append(("filter", instruction, columns, len(self)))
        if len(self) == 0:
            return self
        ds = dataset(self, columns).sem_filter(instruction, depends_on=columns).project(["swan_i"])
        out = ds.run(config()).to_df()
        keep = sorted({int(i) for i in out["swan_i"]}) if len(out) else []
        return self.iloc[keep]

    def sem_map(self, template, suffix="_sem", **kwargs):
        instruction, columns = split_prompt(template)
        state["operators"].append(("map", instruction, columns, len(self)))
        result = self.copy()
        if len(self) == 0:
            result[suffix] = pd.Series(dtype=object)
            return result
        ds = dataset(self, columns).sem_add_columns([{"name": "answer", "type": str, "desc": instruction}],
                                                     depends_on=columns).project(["swan_i", "answer"])
        out = ds.run(config()).to_df()
        got = {int(r["swan_i"]): (None if pd.isna(r["answer"]) else str(r["answer"])) for _, r in out.iterrows()} if len(out) else {}
        result[suffix] = [got.get(i) for i in range(len(self))]
        return result

    pd.DataFrame.sem_filter = sem_filter
    pd.DataFrame.sem_map = sem_map

    frames = {}

    def load(name):
        if name not in frames:
            frames[name] = (pd.read_csv(os.path.join(data_dir, CSV[name])) if name in CSV
                            else pd.read_parquet(os.path.join(data_dir, PARQUET[name])))
        return frames[name].copy()

    spec = importlib.util.spec_from_file_location("hybrid_program", program)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t0 = time.time()
    out = mod.run(load, None)
    seconds = round(time.time() - t0, 2)
    out.to_csv(out_csv, index=False)
    json.dump({"rows": len(out), "seconds": seconds, "operators": state["operators"]}, real_stdout, default=str)
    real_stdout.write("\n")


if __name__ == "__main__":
    main()
