# Q1: Book category classification (semantic_int CASE) cross-joined with decades, LIMIT 5.
import pandas as pd


def run(load, lm):
    books = load("books_info")
    decades = load("decades")

    # semantic_int('...' || title || ' by ' || author) -> sem_map, then parse int and apply CASE
    if len(books) > 0:
        books = books.sem_map(
            "Which category best fits this book? (1) Fiction (2) Non-Fiction (3) Memoir (4) Biography (5) Self-Help. Return only the number: {title} by {author}",
            suffix="_sem",
        )
        cat = pd.to_numeric(books["_sem"].str.extract(r"(\d+)", expand=False), errors="coerce").astype("Int64")
        # CASE WHEN 1..4 ELSE 'Self-Help' (ELSE also covers 5 / unparseable)
        books["category"] = cat.map({1: "Fiction", 2: "Non-Fiction", 3: "Memoir", 4: "Biography"}).fillna("Self-Help")
    else:
        books = books.assign(category=pd.Series(dtype=object))

    books_with_category = books[["title", "category"]]
    out = books_with_category.assign(_ck=1).merge(decades[["decade"]].assign(_ck=1), on="_ck").drop(columns="_ck")
    out = out[["title", "decade", "category"]]
    # LIMIT 5 without ORDER BY -> head in current order
    return out.head(5)
