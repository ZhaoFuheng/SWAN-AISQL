-- File: synth_bookreview_q7.sql
-- Label: Q8
-- Question: Which verified 2017 five-star review praises Make: Electronics for guiding beginners through experiments, warns that the parts kits are pricey, and earned at least 50 helpful votes?

-- Which verified 2017 five-star review praises Make: Electronics for guiding beginners through experiments, warns that the parts kits are pricey, and earned at least 50 helpful votes?

DROP TABLE IF EXISTS books_info;
DROP TABLE IF EXISTS reviews;

CREATE TABLE books_info AS
SELECT *
FROM read_csv_auto('./test/semantic/agent_bench/dataset/book_review/books_info.csv');

CREATE TABLE reviews AS
SELECT *
FROM read_csv_auto('./test/semantic/agent_bench/dataset/book_review/reviews.csv');

WITH
candidates AS (
	SELECT
		CAST(SPLIT_PART(b.book_id, '_', 2) AS INTEGER) AS book_idx,
		b.book_id,
		b.title
	FROM books_info b
	WHERE b.title IS NOT NULL
		AND SEMANTIC('Confirm this is the second edition of Make: Electronics, the hands-on beginner''s electronics guide from the Make magazine team that highlights colorful diagrams and step-by-step experiments.
Title: {b.title}
Subtitle: {b.subtitle}
Author: {b.author}
Categories: {b.categories}
Description: {b.description}
Features: {b.features}
Details: {b.details}')
),

reviews_filtered AS (
	SELECT
		CAST(SPLIT_PART(r.purchase_id, '_', 2) AS INTEGER) AS purchase_idx,
		r.purchase_id,
		r.review_time,
		CAST(r.rating AS DOUBLE) AS rating,
		r.text,
		r.helpful_vote,
		r.verified_purchase
	FROM reviews r
	WHERE r.verified_purchase = 1
		AND CAST(r.rating AS DOUBLE) >= 5
		AND r.helpful_vote >= 50
		AND r.review_time >= TIMESTAMP '2017-01-01'
		AND r.review_time < TIMESTAMP '2018-01-01'
)

SELECT
	c.book_id,
	c.title,
	rf.review_time,
	rf.rating,
	rf.helpful_vote,
	rf.purchase_id,
	rf.text
FROM candidates c
JOIN reviews_filtered rf
	ON rf.purchase_idx = c.book_idx
ORDER BY rf.review_time DESC;
