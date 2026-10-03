-- setup (DuckDB backend)
/* File: synth_bookreview_q5.sql */ /* Label: Q5 */ /* Question: BookReview Q5: Identify the 2020 verified 5-star review praising Cleo Porter’s pandemic mission. */ /* BookReview Q5: Identify the 2020 verified 5-star review praising Cleo Porter’s pandemic mission. */ DROP TABLE IF EXISTS books_info;
CREATE TABLE books_info AS SELECT * FROM READ_CSV_AUTO('./dataset/book_review/books_info.csv', maximum_line_size = 1048576);
DROP TABLE IF EXISTS reviews;
CREATE TABLE reviews AS SELECT * FROM READ_CSV_AUTO('./dataset/book_review/reviews.csv', maximum_line_size = 1048576);
-- query (BlendSQL)
WITH narrowed_books AS (
  SELECT
    b.book_id,
    b.title,
    SUBSTRING(b.description, 1, 600) AS description_snippet,
    b.categories
  FROM books_info AS b
  WHERE
    NOT b.title IS NULL
    AND LOWER(b.title) LIKE '%cleo porter%'
    AND LOWER(COALESCE(b.categories, '')) ILIKE '%science fiction%'
), __b1 AS (
  SELECT
    nb.book_id AS nb__book_id,
    nb.title AS nb__title,
    nb.categories AS nb__categories,
    CONCAT(
      'Title: ',
      CAST(nb.title AS TEXT),
      '
Categories: ',
      CAST(nb.categories AS TEXT),
      '
Description Sample: ',
      CAST(nb.description_snippet AS TEXT),
      '
Answer true or false only.'
    ) AS __p0
  FROM narrowed_books AS nb
), semantic_match AS (
  SELECT
    __b1.nb__book_id AS book_id,
    __b1.nb__title AS title,
    __b1.nb__categories AS categories
  FROM __b1
  WHERE
    {{LLMMap('Return true if this book is a middle-grade science fiction adventure set during a pandemic lockdown in which Cleo Porter leaves a sealed apartment to deliver life-saving medicine, clearly positioned for young readers.', __b1.__p0)}} = TRUE
), indexed_book AS (
  SELECT
    CAST(SPLIT_PART(sm.book_id, '_', 2) AS INT) AS book_idx,
    sm.book_id,
    sm.title
  FROM semantic_match AS sm
), qualified_reviews AS (
  SELECT
    CAST(SPLIT_PART(r.purchase_id, '_', 2) AS INT) AS purchase_idx,
    r.purchase_id,
    r.review_time,
    CAST(r.rating AS DOUBLE) AS rating,
    r.helpful_vote,
    r.text
  FROM reviews AS r
  WHERE
    r.verified_purchase = 1
    AND CAST(r.rating AS DOUBLE) = 5
    AND r.helpful_vote >= 1
    AND r.review_time BETWEEN CAST('2020-01-01 00:00:00' AS TIMESTAMP) AND CAST('2020-12-31 23:59:59' AS TIMESTAMP)
)
SELECT
  ib.book_id,
  ib.title,
  qr.review_time,
  qr.rating,
  qr.helpful_vote,
  qr.purchase_id,
  qr.text
FROM indexed_book AS ib
JOIN qualified_reviews AS qr
  ON qr.purchase_idx = ib.book_idx
ORDER BY
  qr.helpful_vote DESC,
  qr.review_time DESC
