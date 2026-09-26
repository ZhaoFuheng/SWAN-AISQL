# Q6: Verified 2020 five-star (helpful_vote >= 2) review of the Buddy the Soldier Bear book.
import pandas as pd


def run(load, lm):
    books = load("books_info")
    reviews = load("reviews")

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

    out = cand.merge(rf, left_on="book_idx", right_on="purchase_idx")
    out = out[["book_id", "title", "review_time", "rating", "helpful_vote", "purchase_id", "text"]]
    out = out.sort_values("review_time", ascending=False)
    return out
