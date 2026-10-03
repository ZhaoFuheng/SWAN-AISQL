-- setup (DuckDB backend)
/* File: synth_bookreview_q6_cost.sql */ /* Label: Q7 */ /* Question: Which verified 2020 five-star review (with at least 2 helpful votes) matches the Buddy the Soldier Bear context about toddler bedtime comfort and military/veteran support? */ DROP TABLE IF EXISTS books_info;
DROP TABLE IF EXISTS reviews;
CREATE TABLE books_info AS SELECT * FROM READ_CSV_AUTO('./dataset/book_review/books_info.csv');
CREATE TABLE reviews AS SELECT * FROM READ_CSV_AUTO('./dataset/book_review/reviews.csv');
CREATE TABLE rc AS SELECT * FROM READ_CSV_AUTO('./dataset/book_review/review_context.csv');
-- query (BlendSQL)
WITH __b1 AS (
  SELECT
    b.book_id AS b__book_id,
    b.title AS b__title,
    CONCAT(
      'Title: ',
      CAST(b.title AS TEXT),
      '
Subtitle: ',
      CAST(b.subtitle AS TEXT),
      '
Author: ',
      CAST(b.author AS TEXT),
      '
Categories: ',
      CAST(b.categories AS TEXT),
      '
Description: ',
      CAST(b.description AS TEXT),
      '
Features: ',
      CAST(b.features AS TEXT),
      '
Details: ',
      CAST(b.details AS TEXT)
    ) AS __p0
  FROM books_info AS b
  WHERE
    NOT b.title IS NULL
), candidates AS (
  SELECT
    CAST(SPLIT_PART(__b1.b__book_id, '_', 2) AS INT) AS book_idx,
    __b1.b__book_id AS book_id,
    __b1.b__title AS title
  FROM __b1
  WHERE
    {{LLMMap('Confirm this is the Buddy the Soldier Bear children\'s picture book that follows a stuffed bear from a toy store to a battlefield, emphasizes supporting soldiers and veteran families, and is categorized under children\'s literature.', __b1.__p0)}} = TRUE
), reviews_filtered AS (
  SELECT
    CAST(SPLIT_PART(r.purchase_id, '_', 2) AS INT) AS purchase_idx,
    r.purchase_id,
    r.review_time,
    CAST(r.rating AS DOUBLE) AS rating,
    r.text,
    r.helpful_vote,
    r.verified_purchase
  FROM reviews AS r
  WHERE
    r.verified_purchase = 1
    AND CAST(r.rating AS DOUBLE) >= 5
    AND r.helpful_vote >= 2
    AND r.review_time >= CAST('2020-01-01' AS TIMESTAMP)
    AND r.review_time < CAST('2021-01-01' AS TIMESTAMP)
), __b2 AS (
  SELECT
    *,
    CONCAT(
      CAST(rc.context_id AS TEXT),
      '; Name: ',
      CAST(rc.context_name AS TEXT),
      '; Audience: ',
      CAST(rc.audience AS TEXT),
      '; Focus: ',
      CAST(rc.message_focus AS TEXT),
      '; Rule: ',
      CAST(rc.quality_rule AS TEXT),
      '; Does this context match reviews about Buddy the Soldier Bear as toddler bedtime comfort + military/veteran support in 2020 with verified 5-star and >=2 helpful votes? Answer YES or NO. If unsure, answer NO.'
    ) AS __p1
  FROM rc
), filtered_rc AS (
  SELECT
    *
    EXCLUDE (__p1)
  FROM __b2
  WHERE
    {{LLMMap('Context ID:', __b2.__p1)}} = TRUE
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
  FROM candidates AS c
  JOIN reviews_filtered AS rf
    ON rf.purchase_idx = c.book_idx
) AS br
CROSS JOIN filtered_rc
ORDER BY
  br.review_time DESC
