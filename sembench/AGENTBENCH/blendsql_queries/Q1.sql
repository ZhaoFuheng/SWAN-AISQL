-- setup (DuckDB backend)
/* File: proj_opt_q1_book_summary.sql */ /* Label: Q1 */ /* Question: Projection Optimization Query 1: Book Summary with Decades Expansion */ /* Projection Optimization Query 1: Book Summary with Decades Expansion */ /* This query demonstrates semantic projection pull-up optimization: */ /* - 200 books × 13 decades = 2600 potential rows */ /* - semantic_string generates a summary for each book */ /* - LIMIT 5 reduces output to 5 rows */ /* Without optimization: 200 LLM calls (all books get summary) */ /* With optimization: 5 LLM calls (only final 5 rows need summary) */ /* Expected speedup: ~40x */ CREATE TABLE books_info AS SELECT * FROM READ_CSV_AUTO('./dataset/book_review/books_info.csv');
CREATE TABLE decades AS SELECT * FROM READ_CSV_AUTO('./dataset/book_review/decades.csv');
-- query (BlendSQL)
WITH __b1 AS (
  /* Subquery: semantic projection on 200 books */ /* Classify into one of 5 fixed categories instead of generating free text */
  SELECT
    title AS c__title,
    CONCAT(CAST(title AS TEXT), ' by ', CAST(author AS TEXT)) AS __p0
  FROM books_info
), __d1 AS (
  SELECT
    __b1.c__title AS title,
    CASE COALESCE({{LLMMap('Which category best fits this book? (1) Fiction (2) Non-Fiction (3) Memoir (4) Biography (5) Self-Help. Return only the number:', __b1.__p0, return_type='int')}}, 0)
      WHEN 1
      THEN 'Fiction'
      WHEN 2
      THEN 'Non-Fiction'
      WHEN 3
      THEN 'Memoir'
      WHEN 4
      THEN 'Biography'
      ELSE 'Self-Help'
    END AS category
  FROM __b1
)
/* Query: Classify books by genre, cross join with decades, limit to 5 */
SELECT
  title,
  decade,
  category
FROM __d1 AS books_with_category
CROSS JOIN decades
LIMIT 5
