-- File: proj_opt_q2_business_rating.sql
-- Label: Q2
-- Question: Projection Optimization Query 2: Business Quality Rating

-- Projection Optimization Query 2: Business Quality Rating
--
-- This query demonstrates semantic projection pull-up optimization:
-- - 100 businesses × 10 expansion = 1000 potential rows
-- - semantic_int rates business quality 1-10 based on description
-- - LIMIT 3 reduces output to 3 rows
--
-- Without optimization: 100 LLM calls (all businesses get rated)
-- With optimization: 3 LLM calls (only final 3 rows need rating)
-- Expected speedup: ~33x

DROP TABLE IF EXISTS yelp_business;
CREATE TABLE yelp_business AS
SELECT *
FROM read_csv_auto('./test/semantic/agent_bench/dataset/yelp/yelp_business_csv/business.csv', maximum_line_size=1048576);

-- Expansion table
CREATE TABLE expansion(exp_id INT);
INSERT INTO expansion VALUES (1), (2), (3), (4), (5), (6), (7), (8), (9), (10);

-- Query: Classify business quality tier, cross join with expansion, limit to 3
SELECT name, exp_id, quality_tier
FROM (
    -- Subquery: semantic projection on 100 businesses
    SELECT name, description,
           CASE semantic_int('Classify business quality: (1) Poor (2) Fair (3) Good (4) Excellent. Return only number: ' || name || ' - ' || description)
               WHEN 1 THEN 'Poor'
               WHEN 2 THEN 'Fair'
               WHEN 3 THEN 'Good'
               ELSE 'Excellent'
           END as quality_tier
    FROM yelp_business
) businesses_with_score
CROSS JOIN expansion
LIMIT 3;
