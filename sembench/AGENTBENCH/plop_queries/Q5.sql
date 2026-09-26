-- File: synth_bookreview_q5.sql
-- Label: Q5
-- Question: BookReview Q5: Identify the 2020 verified 5-star review praising Cleo Porter’s pandemic mission.

-- BookReview Q5: Identify the 2020 verified 5-star review praising Cleo Porter’s pandemic mission.
DROP TABLE IF EXISTS books_info;
CREATE TABLE books_info AS
SELECT *
FROM read_csv_auto('./test/semantic/agent_bench/dataset/book_review/books_info.csv', maximum_line_size=1048576);

DROP TABLE IF EXISTS reviews;
CREATE TABLE reviews AS
SELECT *
FROM read_csv_auto('./test/semantic/agent_bench/dataset/book_review/reviews.csv', maximum_line_size=1048576);

WITH narrowed_books AS (
	SELECT
		b.book_id,
		b.title,
		substring(b.description, 1, 600) AS description_snippet,
		b.categories
	FROM books_info b
	WHERE b.title IS NOT NULL
		AND lower(b.title) LIKE '%cleo porter%'
		AND lower(coalesce(b.categories, '')) ILIKE '%science fiction%'
),
semantic_match AS (
	SELECT
		nb.book_id,
		nb.title,
		nb.categories
	FROM narrowed_books nb
	WHERE SEMANTIC('Return true if this book is a middle-grade science fiction adventure set during a pandemic lockdown in which Cleo Porter leaves a sealed apartment to deliver life-saving medicine, clearly positioned for young readers.
Title: {nb.title}
Categories: {nb.categories}
Description Sample: {nb.description_snippet}
Answer true or false only.')
),
indexed_book AS (
	SELECT
		CAST(SPLIT_PART(sm.book_id, '_', 2) AS INTEGER) AS book_idx,
		sm.book_id,
		sm.title
	FROM semantic_match sm
),
qualified_reviews AS (
	SELECT
		CAST(SPLIT_PART(r.purchase_id, '_', 2) AS INTEGER) AS purchase_idx,
		r.purchase_id,
		r.review_time,
		CAST(r.rating AS DOUBLE) AS rating,
		r.helpful_vote,
		r.text
	FROM reviews r
	WHERE r.verified_purchase = 1
		AND CAST(r.rating AS DOUBLE) = 5
		AND r.helpful_vote >= 1
		AND r.review_time BETWEEN TIMESTAMP '2020-01-01 00:00:00' AND TIMESTAMP '2020-12-31 23:59:59'
)
SELECT
	ib.book_id,
	ib.title,
	qr.review_time,
	qr.rating,
	qr.helpful_vote,
	qr.purchase_id,
	qr.text
FROM indexed_book ib
JOIN qualified_reviews qr ON qr.purchase_idx = ib.book_idx
ORDER BY qr.helpful_vote DESC, qr.review_time DESC;
