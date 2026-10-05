# Q23: Businesses with explicit category (semantic); individual Jan-Jun 2016 reviews
# with tokenized category and rating.
import pandas as pd

_PREFIX = (
    r"(?:services(?: and products)? in|services including|services, including|including|"
    r"destination for|seeking|specializes in|such as|in the fields of|selection of|range of|"
    r"array of|variety of|mix of|collection of|menu featuring|options for|dedicated to|catering to)"
)

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

    sc = yb
    if len(sc) > 0:
        sc = sc.sem_filter(
            """Return true if the description makes the primary business category explicit (for example, Italian restaurant, dental clinic, yoga studio). Favor entries where you can clearly read at least one category phrase.
Business ID: {business_id}
Name: {name}
Attributes: {attributes}
Description: {description}
Answer true or false only."""
        )
    sc = sc[["business_id", "name", "attributes", "description"]]

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

    rl = yr[yr["business_ref"].str.startswith("businessref_", na=False)].copy()
    rl["business_id"] = "businessid_" + rl["business_ref"].str.extract(r"([0-9]+)$", expand=False)
    rl["review_ts"] = _parse_ts(rl["date"], _REVIEW_FORMATS)
    rl = rl[["business_id", "rating", "review_ts"]]

    out = sc.merge(rl, on="business_id")
    out = out.merge(category_tokens, on="business_id")
    out = out[
        out["review_ts"].notna()
        & (out["review_ts"] >= pd.Timestamp("2016-01-01 00:00:00"))
        & (out["review_ts"] <= pd.Timestamp("2016-06-30 23:59:59"))
    ]
    out = out.rename(columns={"name": "business_name", "review_ts": "review_time"})
    out = out.sort_values(["rating", "review_time"], ascending=[False, False])
    return out[["business_name", "category", "review_time", "rating"]]
