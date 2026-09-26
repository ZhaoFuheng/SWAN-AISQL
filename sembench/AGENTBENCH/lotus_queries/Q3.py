# Q3: Business location type (semantic_int CASE) cross-joined with 20-row expansion, LIMIT 4.
import pandas as pd


def run(load, lm):
    bd = load("gl_business")
    # Expansion table built inline (SQL generate_series(1, 20))
    exp = pd.DataFrame({"e": range(1, 21)})

    if len(bd) > 0:
        bd = bd.sem_map(
            "Classify location: (1) Urban (2) Suburban (3) Rural. Return only number: {name} in {state}. Description: {description}",
            suffix="_sem",
        )
        loc = pd.to_numeric(bd["_sem"].str.extract(r"(\d+)", expand=False), errors="coerce").astype("Int64")
        bd["location_type"] = loc.map({1: "Urban", 2: "Suburban"}).fillna("Rural")
    else:
        bd = bd.assign(location_type=pd.Series(dtype=object))

    tagged = bd[["name", "state", "location_type"]]
    out = tagged.assign(_ck=1).merge(exp.assign(_ck=1), on="_ck").drop(columns="_ck")
    out = out[["name", "e", "location_type"]]
    return out.head(4)
