# Q2: Business quality tier (semantic_int CASE) cross-joined with 10-row expansion, LIMIT 3.
import pandas as pd


def run(load, lm):
    yb = load("yelp_business")
    # Expansion table built inline (SQL CREATE TABLE expansion VALUES 1..10)
    expansion = pd.DataFrame({"exp_id": range(1, 11)})

    if len(yb) > 0:
        yb = yb.sem_map(
            "Classify business quality: (1) Poor (2) Fair (3) Good (4) Excellent. Return only number: {name} - {description}",
            suffix="_sem",
        )
        tier = pd.to_numeric(yb["_sem"].str.extract(r"(\d+)", expand=False), errors="coerce").astype("Int64")
        yb["quality_tier"] = tier.map({1: "Poor", 2: "Fair", 3: "Good"}).fillna("Excellent")
    else:
        yb = yb.assign(quality_tier=pd.Series(dtype=object))

    businesses_with_score = yb[["name", "description", "quality_tier"]]
    out = businesses_with_score.assign(_ck=1).merge(expansion.assign(_ck=1), on="_ck").drop(columns="_ck")
    out = out[["name", "exp_id", "quality_tier"]]
    return out.head(3)
