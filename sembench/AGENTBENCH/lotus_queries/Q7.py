# Q7: Same as Q6 plus a semantic filter over review_context, cross-joined (only br columns selected).
import pandas as pd


def run(load, lm):
    books = load("books_info")
    reviews = load("reviews")
    rc = load("review_context")

    cand = books[books["title"].notna()]
    if len(cand) > 0:
        cand = cand.sem_filter(
            """Confirm this is the Buddy the Soldier Bear children's picture book that follows a stuffed bear from a toy store to a battlefield, emphasizes supporting soldiers and veteran families, and is categorized under children's literature.
Title: {title}
Subtitle: {subtitle}
Author: {author}
Categories: {categories}
Description: {description}
Features: {features}
Details: {details}"""
        )
    cand = cand.assign(book_idx=pd.to_numeric(cand["book_id"].str.split("_").str[1], errors="coerce").astype("Int64"))
    cand = cand[["book_idx", "book_id", "title"]]

    r = reviews.copy()
    r_time = pd.to_datetime(r["review_time"])
    rf = r[
        (r["verified_purchase"] == 1)
        & (r["rating"].astype(float) >= 5)
        & (r["helpful_vote"] >= 2)
        & (r_time >= pd.Timestamp("2020-01-01"))
        & (r_time < pd.Timestamp("2021-01-01"))
    ].copy()
    rf["purchase_idx"] = pd.to_numeric(rf["purchase_id"].str.split("_").str[1], errors="coerce").astype("Int64")
    rf["rating"] = rf["rating"].astype(float)
    rf = rf[["purchase_idx", "purchase_id", "review_time", "rating", "helpful_vote", "text"]]

    br = cand.merge(rf, left_on="book_idx", right_on="purchase_idx")
    br = br[["book_id", "title", "review_time", "rating", "helpful_vote", "purchase_id", "text"]]

    # filtered_rc: semantic filter over the 10-row review_context table
    frc = rc
    if len(frc) > 0:
        frc = frc.sem_filter(
            "Context ID: {context_id}; Name: {context_name}; Audience: {audience}; Focus: {message_focus}; Rule: {quality_rule}; Does this context match reviews about Buddy the Soldier Bear as toddler bedtime comfort + military/veteran support in 2020 with verified 5-star and >=2 helpful votes? Answer YES or NO. If unsure, answer NO."
        )

    # CROSS JOIN filtered_rc (duplicates br rows per matching context; only br columns selected)
    out = br.assign(_ck=1).merge(frc[["context_id"]].assign(_ck=1), on="_ck").drop(columns="_ck")
    out = out[["book_id", "title", "review_time", "rating", "helpful_vote", "purchase_id", "text"]]
    out = out.sort_values("review_time", ascending=False)
    return out
