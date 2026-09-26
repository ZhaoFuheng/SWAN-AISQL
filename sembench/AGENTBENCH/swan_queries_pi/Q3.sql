-- File: proj_opt_q3_location_tag.sql
-- Label: Q3
-- Question: Projection Optimization Query 3: Business Location Tagging

-- Projection Optimization Query 3: Business Location Tagging
--
-- This query demonstrates semantic projection pull-up optimization:
-- - 79 businesses × 20 expansion = 1580 potential rows
-- - semantic_string generates location tag for each business
-- - LIMIT 4 reduces output to 4 rows
--
-- Without optimization: 79 LLM calls (all businesses get tagged)
-- With optimization: 4 LLM calls (only final 4 rows need tagging)
-- Expected speedup: ~20x

DROP TABLE IF EXISTS business_description;
CREATE TABLE business_description AS
SELECT *
FROM read_csv_auto('./dataset/googlelocal/business_description.csv');

-- Expansion table (20 rows)
CREATE TABLE exp(e INT);
INSERT INTO exp SELECT * FROM generate_series(1, 20);

-- Query: Classify business location type, cross join, limit to 4
SELECT name, e, location_type
FROM (
    -- Subquery: semantic projection on 79 businesses
    SELECT name, state, description,
           CASE COALESCE(TRY_CAST(regexp_extract(ai_complete('Classify location: (1) Urban (2) Suburban (3) Rural. Return only number: ' || name || ' in ' || state || '. Description: ' || description || e'\n Return a single integer, do not contain any other words.'), '^\s*([-+]?[0-9]+)', 1) AS INTEGER), 0)
               WHEN 1 THEN 'Urban'
               WHEN 2 THEN 'Suburban'
               ELSE 'Rural'
           END as location_type
    FROM business_description
) tagged_businesses
CROSS JOIN exp
LIMIT 4;
