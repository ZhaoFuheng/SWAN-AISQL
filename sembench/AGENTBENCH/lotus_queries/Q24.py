# Q24: Reviews (2016+) by users who joined in 2016, joined to tokenized categories of
# businesses whose descriptions list categories (semantic).
import pandas as pd

_PREFIX = (
    r"(?:services(?: and products)? in|services including|services, including|including|"
    r"destination for|seeking|specializes in|such as|in the fields of|selection of|range of|"
    r"array of|variety of|mix of|collection of|menu featuring|options for|dedicated to|catering to)"
)

_USER_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%d %b %Y, %H:%M",
    "%d %B %Y, %H:%M",
    "%B %d, %Y at %I:%M %p",
    "%d %b %Y at %I:%M %p",
    "%d %B %Y at %I:%M %p",
]

_REVIEW_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%d %b %Y, %H:%M",
    "%d %b %Y %H:%M",
    "%d %b %Y, %I:%M %p",
    "%d %B %Y, %H:%M",
    "%d %B %Y, %I:%M %p",
    "%B %d, %Y at %I:%M %p",
    "%B %d, %Y %H:%M %p",
    "%d %B %Y at %I:%M %p",
    "%d %b %Y at %I:%M %p",
]


def _parse_ts(s, formats):
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for f in formats:
        out = out.fillna(pd.to_datetime(s, format=f, errors="coerce"))
    return out


def run(load, lm):
    yb = load("yelp_business")
    yr = load("yelp_review")
    yu = load("yelp_user")

    joined_ts = _parse_ts(yu["yelping_since"], _USER_FORMATS)
    qualified_users = yu.loc[
        (joined_ts >= pd.Timestamp("2016-01-01 00:00:00"))
        & (joined_ts <= pd.Timestamp("2016-12-31 23:59:59")),
        ["user_id"],
    ]

    pr = yr[yr["business_ref"].str.startswith("businessref_", na=False)].copy()
    pr["business_id"] = "businessid_" + pr["business_ref"].str.extract(r"([0-9]+)$", expand=False)
    pr["review_ts"] = _parse_ts(pr["date"], _REVIEW_FORMATS)
    pr = pr[["business_id", "user_id", "review_ts"]]

    fr = pr.merge(qualified_users, on="user_id")
    fr = fr[fr["review_ts"].notna() & (fr["review_ts"] >= pd.Timestamp("2016-01-01 00:00:00"))]

    active_businesses = fr[["business_id"]].drop_duplicates()

    sc = yb.merge(active_businesses, on="business_id")
    if len(sc) > 0:
        sc = sc.sem_filter(
            """Return true if the business description clearly lists multiple categories or service types that can be tokenized. Focus on entries with explicit category phrases.
Business ID: {business_id}
Attributes: {attributes}
Description: {description}
Respond with true or false only."""
        )
    sc = sc[["business_id", "attributes", "description"]]

    raw = sc["description"].str.extract(_PREFIX + r"\s+(.+?)(?:\.|$)", expand=False).fillna("")
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

    out = category_tokens.merge(fr, on="business_id")
    out = out.sort_values("review_ts", ascending=False)
    return out[["category", "user_id", "review_ts"]]
