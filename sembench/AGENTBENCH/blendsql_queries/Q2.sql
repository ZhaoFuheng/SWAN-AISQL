-- setup (DuckDB backend)
/* File: proj_opt_q2_business_rating.sql */ /* Label: Q2 */ /* Question: Projection Optimization Query 2: Business Quality Rating */ /* Projection Optimization Query 2: Business Quality Rating */ /* This query demonstrates semantic projection pull-up optimization: */ /* - 100 businesses × 10 expansion = 1000 potential rows */ /* - semantic_int rates business quality 1-10 based on description */ /* - LIMIT 3 reduces output to 3 rows */ /* Without optimization: 100 LLM calls (all businesses get rated) */ /* With optimization: 3 LLM calls (only final 3 rows need rating) */ /* Expected speedup: ~33x */ DROP TABLE IF EXISTS yelp_business;
CREATE TABLE yelp_business AS SELECT * FROM READ_CSV_AUTO('./dataset/yelp/yelp_business_csv/business.csv', maximum_line_size = 1048576);
/* Expansion table */ CREATE TABLE expansion (exp_id INT);
INSERT INTO expansion VALUES (1), (2), (3), (4), (5), (6), (7), (8), (9), (10);
-- query (BlendSQL)
WITH __b1 AS (
  /* Subquery: semantic projection on 100 businesses */
  SELECT
    name AS c__name,
    description AS c__description,
    CONCAT(CAST(name AS TEXT), ' - ', CAST(description AS TEXT)) AS __p0
  FROM yelp_business
), __d1 AS (
  SELECT
    __b1.c__name AS name,
    __b1.c__description AS description,
    CASE COALESCE({{LLMMap('Classify business quality: (1) Poor (2) Fair (3) Good (4) Excellent. Return only number:', __b1.__p0, return_type='int')}}, 0)
      WHEN 1
      THEN 'Poor'
      WHEN 2
      THEN 'Fair'
      WHEN 3
      THEN 'Good'
      ELSE 'Excellent'
    END AS quality_tier
  FROM __b1
)
/* Query: Classify business quality tier, cross join with expansion, limit to 3 */
SELECT
  name,
  exp_id,
  quality_tier
FROM __d1 AS businesses_with_score
CROSS JOIN expansion
LIMIT 3
