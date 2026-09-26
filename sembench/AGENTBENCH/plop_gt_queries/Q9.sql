-- File: synth_bookreview_q8.sql
-- Label: Q9
-- Question: Which verified 2013 five-star review raves about the memoir French Illusions, highlighting the au pair’s clashes with Madame Dubois in the Loire Valley and earning at least 20 helpful votes?

-- Which verified 2013 five-star review raves about the memoir French Illusions, highlighting the au pair’s clashes with Madame Dubois in the Loire Valley and earning at least 20 helpful votes?

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
		AND SEMANTIC('Confirm this is French Illusions, the travel memoir where Linda Kovic-Skow recounts her au pair year with the demanding Dubois family in France''s Loire Valley.
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
		AND r.helpful_vote >= 20
		AND r.review_time >= TIMESTAMP '2013-01-01'
		AND r.review_time < TIMESTAMP '2014-01-01'
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
