-- File: synth_bookreview_q6_cost.sql
-- Label: Q7
-- Question: Which verified 2020 five-star review (with at least 2 helpful votes) matches the Buddy the Soldier Bear context about toddler bedtime comfort and military/veteran support?

DROP TABLE IF EXISTS books_info;
DROP TABLE IF EXISTS reviews;

CREATE TABLE books_info AS
SELECT *
FROM read_csv_auto('./dataset/book_review/books_info.csv');

CREATE TABLE reviews AS
SELECT *
FROM read_csv_auto('./dataset/book_review/reviews.csv');

CREATE TABLE rc AS
SELECT *
FROM read_csv_auto('./dataset/book_review/review_context.csv');



WITH
candidates AS (
	SELECT
		CAST(SPLIT_PART(b.book_id, '_', 2) AS INTEGER) AS book_idx,
		b.book_id,
		b.title
	FROM books_info b
	WHERE b.title IS NOT NULL
		AND (lower(ai_complete('Confirm this is the Buddy the Soldier Bear children''s picture book that follows a stuffed bear from a toy store to a battlefield, emphasizes supporting soldiers and veteran families, and is categorized under children''s literature.
Title: ' || b.title || '
Subtitle: ' || b.subtitle || '
Author: ' || b.author || '
Categories: ' || b.categories || '
Description: ' || b.description || '
Features: ' || b.features || '
Details: ' || b.details || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
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
		AND r.helpful_vote >= 2
		AND r.review_time >= TIMESTAMP '2020-01-01'
		AND r.review_time < TIMESTAMP '2021-01-01'
),
filtered_rc AS (
SELECT *
FROM rc
WHERE (lower(ai_complete('Context ID: ' || rc.context_id || '; Name: ' || rc.context_name || '; Audience: ' || rc.audience || '; Focus: ' || rc.message_focus || '; Rule: ' || rc.quality_rule || '; Does this context match reviews about Buddy the Soldier Bear as toddler bedtime comfort + military/veteran support in 2020 with verified 5-star and >=2 helpful votes? Answer YES or NO. If unsure, answer NO.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
)

SELECT
	br.book_id,
	br.title,
	br.review_time,
	br.rating,
	br.helpful_vote,
	br.purchase_id,
	br.text
FROM (
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
) br
CROSS JOIN filtered_rc
ORDER BY br.review_time DESC;
