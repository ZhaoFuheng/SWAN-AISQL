-- setup (DuckDB backend)
/* File: synth_bookreview_q4.sql */ /* Label: Q4 */ /* Question: Which post-2016 books are semantically judged as recently popular, and what are their positive reviews (rating >= 4) with review time, purchase id, and title? */ DROP TABLE IF EXISTS books_info;
DROP TABLE IF EXISTS reviews;
CREATE TABLE books_info AS SELECT * FROM READ_CSV_AUTO('./dataset/book_review/books_info.csv');
CREATE TABLE reviews AS SELECT * FROM READ_CSV_AUTO('./dataset/book_review/reviews.csv');
-- query (BlendSQL)
WITH post2016_books AS (
  /* candidate book_ids that have at least one review in/after 2016 */
  SELECT DISTINCT
    'bookid_' || REPLACE(r.purchase_id, 'purchaseid_', '') AS book_id
  FROM reviews AS r
  WHERE
    EXTRACT(YEAR FROM r.review_time) >= 2016
), __b1 AS (
  /* semantic boolean filter on metadata fields for post-2016 candidates (used as WHERE filter) */
  SELECT
    b.book_id AS b__book_id,
    b.title AS b__title,
    p.book_id AS p__book_id,
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
      CAST(b.description AS TEXT)
    ) AS __p0
  FROM books_info AS b
  JOIN post2016_books AS p
    ON p.book_id = b.book_id
  WHERE
    NOT b.title IS NULL
), sem_popular AS (
  SELECT DISTINCT
    __b1.b__book_id AS book_id,
    __b1.b__title AS title
  FROM __b1
  WHERE
    {{LLMMap('Is this book recently popular since 2016?', __b1.__p0)}} = TRUE
)
SELECT
  r.review_time,
  r.rating,
  r.purchase_id,
  sp.book_id,
  sp.title
FROM reviews AS r
JOIN sem_popular AS sp
  ON sp.book_id = 'bookid_' || REPLACE(r.purchase_id, 'purchaseid_', '')
WHERE
  EXTRACT(YEAR FROM r.review_time) >= 2016 AND CAST(r.rating AS INT) >= 4
ORDER BY
  r.review_time DESC
