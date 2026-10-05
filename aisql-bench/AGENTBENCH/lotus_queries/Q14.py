# Q14: Campgrounds (semantic) with 2021 reviews mentioning off-road trails or dirt biking.
import pandas as pd


def run(load, lm):
    bd = load("gl_business")
    r = load("gl_review")

    c = bd[bd["num_of_reviews"].notna()]
    c = c[c["num_of_reviews"].astype(int) >= 5]
    if len(c) > 0:
        # SQL placeholder {bd.misc} -> actual CSV column is spelled MISC
        c = c.sem_filter(
            """Return true only if this place is a campground, RV park, off-road recreation area, or outdoor trailhead. Otherwise return false.
Name: {name}
Description: {description}
Misc: {MISC}
Respond with true or false only."""
        )
    c = c.assign(num_reviews=c["num_of_reviews"].astype(int))
    c = c[["gmap_id", "name", "description", "num_reviews", "state"]]

    pr = c.merge(r, on="gmap_id", suffixes=("", "_r"))
    txt = pr["text"].fillna("")
    low = txt.str.lower()
    pr = pr[
        pr["text"].notna()
        & (txt.str.len() >= 80)
        & (pr["rating"].astype(float) >= 4)
        & pr["time"].fillna("").str.contains("2021", case=False, regex=False)
        & (
            low.str.contains("dirt bike", regex=False)
            | low.str.contains("trail", regex=False)
            | low.str.contains("off-road", regex=False)
            | low.str.contains("camp", regex=False)
        )
    ].copy()

    pr["review_time"] = pr["time"]
    pr["rating"] = pr["rating"].astype(float)
    out = pr[["gmap_id", "name", "num_reviews", "state", "review_time", "rating", "text"]]
    out = out.sort_values(["rating", "review_time"], ascending=[False, False])
    return out
