# Q22: WiFi-enabled businesses (semantic), U.S. state code extracted from description,
# joined with 2018 reviews and tips.
import pandas as pd

_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY",
]


def run(load, lm):
    yb = load("yelp_business")
    yr = load("yelp_review")
    yt = load("yelp_tip")

    sw = yb
    if len(sw) > 0:
        sw = sw.sem_filter(
            """Return true if the business offers public WiFi access for its customers. Use both the attributes and description for evidence.
Business ID: {business_id}
Attributes: {attributes}
Description: {description}
Respond true or false only."""
        )
    sw = sw[["business_id", "attributes", "description"]]

    state_lookup = sw[["business_id"]].copy()
    state_lookup["state_code"] = sw["description"].str.extract(r",\s*([A-Z]{2})\b", expand=False)
    valid_states = state_lookup[state_lookup["state_code"].isin(_STATES)]

    rl = yr[yr["business_ref"].str.startswith("businessref_", na=False)].copy()
    rl["business_id"] = "businessid_" + rl["business_ref"].str.extract(r"([0-9]+)$", expand=False)
    rl = rl.rename(columns={"date": "review_date"})[["business_id", "rating", "review_date"]]

    tl = yt[yt["business_ref"].str.startswith("businessref_", na=False)].copy()
    tl["business_id"] = "businessid_" + tl["business_ref"].str.extract(r"([0-9]+)$", expand=False)
    tl = tl.rename(columns={"date": "tip_date"})[["business_id", "tip_date"]]

    out = valid_states.merge(sw, on="business_id")
    out = out.merge(rl, on="business_id")
    out = out.merge(tl, on="business_id")
    out = out[
        out["attributes"].fillna("").str.contains('"WiFi":', case=False, regex=False)
        & out["review_date"].fillna("").str.contains("2018", case=False, regex=False)
        & out["tip_date"].fillna("").str.contains("2018", case=False, regex=False)
    ]
    out = out.sort_values(["rating", "review_date"], ascending=[False, False])
    out = out.rename(columns={"state_code": "state"})
    return out[["state", "business_id", "review_date", "rating"]]
