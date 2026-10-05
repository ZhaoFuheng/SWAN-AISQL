# Q21: Credit-card-friendly businesses (semantic), category tokens from description,
# joined with 2019 reviews and tips.
import pandas as pd

_PREFIX = (
    r"(?:services(?: and products)? in|services including|services, including|including|"
    r"destination for|seeking|specializes in|such as|in the fields of|selection of|range of|"
    r"array of|variety of|mix of|collection of|menu featuring|options for|dedicated to|catering to)"
)


def run(load, lm):
    yb = load("yelp_business")
    yr = load("yelp_review")
    yt = load("yelp_tip")

    sc = yb
    if len(sc) > 0:
        sc = sc.sem_filter(
            """Return true if the business clearly accepts credit card payments. Prefer explicit evidence from attributes or the narrative.
Business ID: {business_id}
Attributes: {attributes}
Description: {description}
Answer true or false only."""
        )
    sc = sc[["business_id", "attributes", "description"]]

    # normalized_categories: regexp_extract then strip a leading prefix again
    raw = sc["description"].str.extract(_PREFIX + r"\s+(.+?)(?:\.|$)", expand=False).fillna("")
    raw = raw.str.replace("^" + _PREFIX + r"\s+", "", regex=True)

    tmp = sc[["business_id"]].copy()
    tmp["raw"] = (
        raw.str.replace(", and ", ", ", regex=False)
        .str.replace(" and ", ", ", regex=False)
        .str.replace("  ", " ", regex=False)
    )
    tmp["category"] = tmp["raw"].str.split(",")
    tok = tmp.explode("category")
    tok["category"] = tok["category"].fillna("").str.strip().str.lower()
    category_tokens = tok.loc[tok["category"] != "", ["business_id", "category"]]

    rl = yr[yr["business_ref"].str.startswith("businessref_", na=False)].copy()
    rl["business_id"] = "businessid_" + rl["business_ref"].str.extract(r"([0-9]+)$", expand=False)
    rl = rl.rename(columns={"date": "review_date"})[["business_id", "rating", "review_date"]]

    tl = yt[yt["business_ref"].str.startswith("businessref_", na=False)].copy()
    tl["business_id"] = "businessid_" + tl["business_ref"].str.extract(r"([0-9]+)$", expand=False)
    tl = tl.rename(columns={"date": "tip_date"})[["business_id", "tip_date"]]

    out = category_tokens.merge(sc, on="business_id")
    out = out.merge(rl, on="business_id")
    out = out.merge(tl, on="business_id")
    out = out[
        out["attributes"].fillna("").str.contains('"BusinessAcceptsCreditCards": "True"', case=False, regex=False)
        & out["review_date"].fillna("").str.contains("2019", case=False, regex=False)
        & out["tip_date"].fillna("").str.contains("2019", case=False, regex=False)
    ]
    out = out.sort_values(["rating", "review_date"], ascending=[False, False])
    return out[["category", "business_id", "review_date", "rating"]]
