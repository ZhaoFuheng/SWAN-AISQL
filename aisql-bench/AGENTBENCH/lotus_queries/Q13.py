# Q13: Nail salons (semantic) with detailed 2021 reviews highlighting manicure/brow results.
import pandas as pd


def run(load, lm):
    bd = load("gl_business")
    r = load("gl_review")

    n = bd[bd["num_of_reviews"].notna()]
    n = n[n["num_of_reviews"].astype(int) >= 3]
    if len(n) > 0:
        # SQL placeholder {bd.misc} -> actual CSV column is spelled MISC
        n = n.sem_filter(
            """Return true only if this business primarily offers nail salon, manicure, pedicure, brow, or spa services. Otherwise return false.
Name: {name}
Description: {description}
Misc: {MISC}
Respond with true or false only."""
        )
    n = n.assign(num_reviews=n["num_of_reviews"].astype(int))
    n = n[["gmap_id", "name", "description", "num_reviews", "state"]]

    pr = n.merge(r, on="gmap_id", suffixes=("", "_r"))
    txt = pr["text"].fillna("")
    low = txt.str.lower()
    pr = pr[
        pr["text"].notna()
        & (txt.str.len() >= 120)
        & (pr["rating"].astype(float) >= 4.5)
        & pr["time"].fillna("").str.contains("2021", case=False, regex=False)
        & (
            low.str.contains("nail", regex=False)
            | low.str.contains("mani", regex=False)
            | low.str.contains("pedi", regex=False)
            | low.str.contains("brow", regex=False)
            | low.str.contains("dip", regex=False)
        )
    ].copy()

    pr["review_time"] = pr["time"]
    pr["rating"] = pr["rating"].astype(float)
    out = pr[["gmap_id", "name", "num_reviews", "state", "review_time", "rating", "text"]]
    out = out.sort_values(["rating", "review_time"], ascending=[False, False])
    return out
