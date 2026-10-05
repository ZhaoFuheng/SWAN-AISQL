# Q11: Cannabis dispensaries (semantic) with long, high-rated 2021 reviews praising staff.
import pandas as pd


def run(load, lm):
    bd = load("gl_business")
    r = load("gl_review")

    d = bd
    if len(d) > 0:
        # SQL placeholder {bd.misc} -> actual CSV column is spelled MISC
        d = d.sem_filter(
            """Return true only if this business is primarily a cannabis dispensary, delivery service, or weed retailer. Otherwise return false.
Name: {name}
Description: {description}
Misc: {MISC}
Respond with true or false only."""
        )
    d = d.assign(num_reviews=d["num_of_reviews"].astype(int))
    d = d[["gmap_id", "name", "description", "num_reviews", "state"]]

    pr = d.merge(r, on="gmap_id", suffixes=("", "_r"))
    txt = pr["text"].fillna("")
    low = txt.str.lower()
    pr = pr[
        pr["text"].notna()
        & (txt.str.len() >= 160)
        & (pr["rating"].astype(float) >= 4.8)
        & pr["time"].fillna("").str.contains("2021", case=False, regex=False)
        & low.str.contains("staff", regex=False)
        & (
            low.str.contains("helpful", regex=False)
            | low.str.contains("friendly", regex=False)
            | low.str.contains("knowledgeable", regex=False)
            | low.str.contains("budtender", regex=False)
            | low.str.contains("bud tender", regex=False)
        )
    ].copy()

    pr["review_time"] = pr["time"]
    pr["rating"] = pr["rating"].astype(float)
    out = pr[["gmap_id", "name", "num_reviews", "state", "review_time", "rating", "text"]]
    out = out.assign(_len=out["text"].str.len())
    out = out.sort_values(["rating", "review_time", "_len"], ascending=[False, False, False]).drop(columns="_len")
    return out
