-- File: synth_bookreview_q4.sql
-- Label: Q4
-- Question: Which post-2016 books are semantically judged as recently popular, and what are their positive reviews (rating >= 4) with review time, purchase id, and title?

DROP TABLE IF EXISTS books_info;
DROP TABLE IF EXISTS reviews;

CREATE TABLE books_info AS
SELECT * FROM read_csv_auto('./dataset/book_review/books_info.csv');

CREATE TABLE reviews AS
SELECT * FROM read_csv_auto('./dataset/book_review/reviews.csv');

-- NLQ: List individual positive reviews (rating >= 4) since 2016 for books that a semantic classifier marks as "recently popular" when given the book's metadata. Return review_time, rating, purchase_id, book_id and title.

WITH
post2016_books AS (
		-- candidate book_ids that have at least one review in/after 2016
		SELECT DISTINCT 'bookid_' || REPLACE(r.purchase_id, 'purchaseid_', '') AS book_id
		FROM reviews r
		WHERE EXTRACT(YEAR FROM r.review_time) >= 2016
),
sem_popular AS (
		-- semantic boolean filter on metadata fields for post-2016 candidates (used as WHERE filter)
		SELECT DISTINCT b.book_id,
					 b.title
		FROM books_info b CROSS JOIN LATERAL (SELECT COALESCE(CAST(b.title AS VARCHAR), '') AS title, COALESCE(CAST(b.subtitle AS VARCHAR), '') AS subtitle, COALESCE(CAST(b.author AS VARCHAR), '') AS author, COALESCE(CAST(b.categories AS VARCHAR), '') AS categories, COALESCE(CAST(b.description AS VARCHAR), '') AS description) b_text
		JOIN post2016_books p ON p.book_id = b.book_id
		WHERE b.title IS NOT NULL
			AND s'Is this book recently popular since 2016?
				Title: {b_text.title}
				Subtitle: {b_text.subtitle}
				Author: {b_text.author}
				Categories: {b_text.categories}
				Description: {b_text.description}
				'
)

SELECT
		r.review_time,
		r.rating,
		r.purchase_id,
		sp.book_id,
		sp.title
FROM reviews r
JOIN sem_popular sp
	ON sp.book_id = 'bookid_' || REPLACE(r.purchase_id, 'purchaseid_', '')
WHERE EXTRACT(YEAR FROM r.review_time) >= 2016
	AND CAST(r.rating AS INTEGER) >= 4
ORDER BY r.review_time DESC;
