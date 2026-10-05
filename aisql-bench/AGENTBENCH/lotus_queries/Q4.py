# Q4: Positive (rating >= 4) post-2016 reviews of books semantically judged "recently popular".
import pandas as pd


def run(load, lm):
    books = load("books_info")
    reviews = load("reviews")

    # post2016_books: distinct book_id derived from purchase_id for reviews in/after 2016
    rt = pd.to_datetime(reviews["review_time"])
    p = reviews.loc[rt.dt.year >= 2016, ["purchase_id"]].copy()
    p["book_id"] = "bookid_" + p["purchase_id"].str.replace("purchaseid_", "", regex=False)
    post2016_books = p[["book_id"]].drop_duplicates()

    # sem_popular: semantic filter on candidate books
    cand = books.merge(post2016_books, on="book_id")
    cand = cand[cand["title"].notna()]
    if len(cand) > 0:
        cand = cand.sem_filter(
            """Is this book recently popular since 2016?
				Title: {title}
				Subtitle: {subtitle}
				Author: {author}
				Categories: {categories}
				Description: {description}
				"""
        )
    sem_popular = cand[["book_id", "title"]].drop_duplicates()

    r = reviews.copy()
    r_time = pd.to_datetime(r["review_time"])
    r = r[(r_time.dt.year >= 2016) & (r["rating"].astype(int) >= 4)]
    r["book_id"] = "bookid_" + r["purchase_id"].str.replace("purchaseid_", "", regex=False)
    r = r[["review_time", "rating", "purchase_id", "book_id"]]

    out = r.merge(sem_popular, on="book_id")
    out = out[["review_time", "rating", "purchase_id", "book_id", "title"]]
    out = out.sort_values("review_time", ascending=False)
    return out
