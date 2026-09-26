# Q5: The 2020 verified 5-star review praising Cleo Porter's pandemic mission.
import pandas as pd


def run(load, lm):
    books = load("books_info")
    reviews = load("reviews")

    # narrowed_books: relational narrowing before the semantic filter
    nb = books[books["title"].notna()]
    nb = nb[nb["title"].str.lower().str.contains("cleo porter", regex=False)]
    nb = nb[nb["categories"].fillna("").str.lower().str.contains("science fiction", regex=False)]
    nb = nb.assign(description_snippet=nb["description"].str.slice(0, 600))
    nb = nb[["book_id", "title", "description_snippet", "categories"]]

    if len(nb) > 0:
        nb = nb.sem_filter(
            """Return true if this book is a middle-grade science fiction adventure set during a pandemic lockdown in which Cleo Porter leaves a sealed apartment to deliver life-saving medicine, clearly positioned for young readers.
Title: {title}
Categories: {categories}
Description Sample: {description_snippet}
Answer true or false only."""
        )

    ib = nb.assign(book_idx=pd.to_numeric(nb["book_id"].str.split("_").str[1], errors="coerce").astype("Int64"))
    ib = ib[["book_idx", "book_id", "title"]]

    r = reviews.copy()
    r_time = pd.to_datetime(r["review_time"])
    qr = r[
        (r["verified_purchase"] == 1)
        & (r["rating"].astype(float) == 5)
        & (r["helpful_vote"] >= 1)
        & (r_time >= pd.Timestamp("2020-01-01 00:00:00"))
        & (r_time <= pd.Timestamp("2020-12-31 23:59:59"))
    ].copy()
    qr["purchase_idx"] = pd.to_numeric(qr["purchase_id"].str.split("_").str[1], errors="coerce").astype("Int64")
    qr["rating"] = qr["rating"].astype(float)
    qr = qr[["purchase_idx", "purchase_id", "review_time", "rating", "helpful_vote", "text"]]

    out = ib.merge(qr, left_on="book_idx", right_on="purchase_idx")
    out = out[["book_id", "title", "review_time", "rating", "helpful_vote", "purchase_id", "text"]]
    out = out.sort_values(["helpful_vote", "review_time"], ascending=[False, False])
    return out
