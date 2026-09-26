# Q10: Massage/spa businesses (semantic) with detailed high-rated 2021-2024 reviews.
import pandas as pd


def run(load, lm):
    bd = load("gl_business")
    r = load("gl_review")

    mb = bd[bd["num_of_reviews"].notna()]
    mb = mb[mb["num_of_reviews"].astype(int) >= 15]
    mb = mb[mb["state"].str.startswith("Open", na=False)]
    if len(mb) > 0:
        # SQL placeholder {bd.misc} -> actual CSV column is spelled MISC
        mb = mb.sem_filter(
            """Return true if this business primarily offers massage therapy or spa treatments. Otherwise return false.
Name: {name}
Description: {description}
Misc: {MISC}
Number of Reviews: {num_of_reviews}
Respond with true or false only."""
        )
    mb = mb[["gmap_id", "name", "description", "num_of_reviews", "state"]]

    pr = mb.merge(r, on="gmap_id", suffixes=("", "_r"))
    t = pr["time"].fillna("")
    year_ok = (
        t.str.contains("2021", regex=False)
        | t.str.contains("2022", regex=False)
        | t.str.contains("2023", regex=False)
        | t.str.contains("2024", regex=False)
    )
    pr = pr[
        (pr["rating"].astype(float) >= 4.8)
        & pr["text"].notna()
        & (pr["text"].str.len() >= 150)
        & year_ok
    ].copy()

    pr["business_review_count"] = pr["num_of_reviews"].astype(int)
    pr["review_time"] = pr["time"]
    pr["rating"] = pr["rating"].astype(float)
    out = pr[["gmap_id", "name", "business_review_count", "state", "review_time", "rating", "text"]]
    out = out.sort_values(["rating", "review_time"], ascending=[False, False])
    return out
