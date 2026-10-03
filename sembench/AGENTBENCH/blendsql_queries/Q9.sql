-- setup (DuckDB backend)
/* File: synth_bookreview_q8.sql */ /* Label: Q9 */ /* Question: Which verified 2013 five-star review raves about the memoir French Illusions, highlighting the au pair’s clashes with Madame Dubois in the Loire Valley and earning at least 20 helpful votes? */ /* Which verified 2013 five-star review raves about the memoir French Illusions, highlighting the au pair’s clashes with Madame Dubois in the Loire Valley and earning at least 20 helpful votes? */ DROP TABLE IF EXISTS books_info;
DROP TABLE IF EXISTS reviews;
CREATE TABLE books_info AS SELECT * FROM READ_CSV_AUTO('./dataset/book_review/books_info.csv');
CREATE TABLE reviews AS SELECT * FROM READ_CSV_AUTO('./dataset/book_review/reviews.csv');
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
    {{LLMMap('Confirm this is French Illusions, the travel memoir where Linda Kovic-Skow recounts her au pair year with the demanding Dubois family in France\'s Loire Valley.', __b1.__p0)}} = TRUE
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
    AND r.helpful_vote >= 20
    AND r.review_time >= CAST('2013-01-01' AS TIMESTAMP)
    AND r.review_time < CAST('2014-01-01' AS TIMESTAMP)
)
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
ORDER BY
  rf.review_time DESC
