# Q12: Open auto repair shops (semantic) with long 2021 reviews praising honest/quick/fast service.
import pandas as pd


def run(load, lm):
    bd = load("gl_business")
    r = load("gl_review")

    s = bd[bd["num_of_reviews"].notna()]
    s = s[s["num_of_reviews"].astype(int) >= 10]
    # state ILIKE 'Open%' -> case-insensitive startswith
    s = s[s["state"].fillna("").str.lower().str.startswith("open")]
    if len(s) > 0:
        # SQL placeholder {bd.misc} -> actual CSV column is spelled MISC
        s = s.sem_filter(
            """Return true only if this business is primarily an auto repair, mechanic, brake, tire, or vehicle maintenance shop. Otherwise return false.
Name: {name}
Description: {description}
Misc: {MISC}
Respond with true or false only."""
        )
    s = s.assign(num_reviews=s["num_of_reviews"].astype(int))
    s = s[["gmap_id", "name", "description", "num_reviews", "state"]]

    pr = s.merge(r, on="gmap_id", suffixes=("", "_r"))
    txt = pr["text"].fillna("")
    low = txt.str.lower()
    pr = pr[
        pr["text"].notna()
        & (txt.str.len() >= 120)
        & (pr["rating"].astype(float) >= 4.5)
        & pr["time"].fillna("").str.contains("2021", case=False, regex=False)
        & (
            low.str.contains("honest", regex=False)
            | low.str.contains("quick", regex=False)
            | low.str.contains("fast", regex=False)
        )
    ].copy()

    pr["review_time"] = pr["time"]
    pr["rating"] = pr["rating"].astype(float)
    pr["text_length"] = pr["text"].str.len()
    out = pr[["gmap_id", "name", "num_reviews", "state", "review_time", "rating", "text", "text_length"]]
    out = out.sort_values(["rating", "text_length"], ascending=[False, False])
    return out
