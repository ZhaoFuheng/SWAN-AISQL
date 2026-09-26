-- File: proj_opt_q1_book_summary.sql
-- Label: Q1
-- Question: Projection Optimization Query 1: Book Summary with Decades Expansion

-- Projection Optimization Query 1: Book Summary with Decades Expansion
--
-- This query demonstrates semantic projection pull-up optimization:
-- - 200 books × 13 decades = 2600 potential rows
-- - semantic_string generates a summary for each book
-- - LIMIT 5 reduces output to 5 rows
--
-- Without optimization: 200 LLM calls (all books get summary)
-- With optimization: 5 LLM calls (only final 5 rows need summary)
-- Expected speedup: ~40x

CREATE TABLE books_info AS
SELECT *
FROM read_csv_auto('./dataset/book_review/books_info.csv');

CREATE TABLE decades AS
SELECT *
FROM read_csv_auto('./dataset/book_review/decades.csv');

-- Query: Classify books by genre, cross join with decades, limit to 5
SELECT title, decade, category
FROM (
    -- Subquery: semantic projection on 200 books
    -- Classify into one of 5 fixed categories instead of generating free text
    SELECT title, 
           CASE COALESCE(TRY_CAST(regexp_extract(ai_complete('Which category best fits this book? (1) Fiction (2) Non-Fiction (3) Memoir (4) Biography (5) Self-Help. Return only the number: ' || title || ' by ' || author || e'\n Return a single integer, do not contain any other words.'), '^\s*([-+]?[0-9]+)', 1) AS INTEGER), 0)
               WHEN 1 THEN 'Fiction'
               WHEN 2 THEN 'Non-Fiction'
               WHEN 3 THEN 'Memoir'
               WHEN 4 THEN 'Biography'
               ELSE 'Self-Help'
           END as category
    FROM books_info
) books_with_category
CROSS JOIN decades
LIMIT 5;
