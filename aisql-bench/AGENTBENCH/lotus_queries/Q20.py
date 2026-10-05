# Q20: Businesses offering parking (semantic) with 2018 reviews/tips, bike parking + Philadelphia.
import pandas as pd


def run(load, lm):
    yb = load("yelp_business")
    yr = load("yelp_review")
    yt = load("yelp_tip")

    sp = yb
    if len(sp) > 0:
        sp = sp.sem_filter(
            """Return true if the business offers customer parking (lot, garage, street, or valet) or any form of bike parking. Use the provided attributes and description to decide.
Business ID: {business_id}
Attributes: {attributes}
Description: {description}
Respond with true or false only."""
        )
    sp = sp[["business_id", "attributes", "description"]]

    rl = yr[yr["business_ref"].str.startswith("businessref_", na=False)].copy()
    rl["business_id"] = "businessid_" + rl["business_ref"].str.extract(r"([0-9]+)$", expand=False)
    rl = rl.rename(columns={"date": "review_date"})[["business_id", "review_date", "rating"]]

    tl = yt[yt["business_ref"].str.startswith("businessref_", na=False)].copy()
    tl["business_id"] = "businessid_" + tl["business_ref"].str.extract(r"([0-9]+)$", expand=False)
    tl = tl.rename(columns={"date": "tip_date"})[["business_id", "tip_date"]]

    out = sp.merge(rl, on="business_id").merge(tl, on="business_id", how="left")
    out = out[
        out["review_date"].fillna("").str.contains("2018", case=False, regex=False)
        & (out["rating"] >= 4)
        & out["attributes"].fillna("").str.contains('"BikeParking": "True"', case=False, regex=False)
        & out["tip_date"].fillna("").str.contains("2018", case=False, regex=False)
        & out["description"].fillna("").str.contains("Philadelphia", case=False, regex=False)
    ]
    out = out.sort_values(["rating", "review_date"], ascending=[False, False])
    return out[["business_id", "review_date", "rating"]]
