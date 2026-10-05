-- setup (DuckDB backend)
/* File: proj_opt_q3_location_tag.sql */ /* Label: Q3 */ /* Question: Projection Optimization Query 3: Business Location Tagging */ /* Projection Optimization Query 3: Business Location Tagging */ /* This query demonstrates semantic projection pull-up optimization: */ /* - 79 businesses × 20 expansion = 1580 potential rows */ /* - semantic_string generates location tag for each business */ /* - LIMIT 4 reduces output to 4 rows */ /* Without optimization: 79 LLM calls (all businesses get tagged) */ /* With optimization: 4 LLM calls (only final 4 rows need tagging) */ /* Expected speedup: ~20x */ DROP TABLE IF EXISTS business_description;
CREATE TABLE business_description AS SELECT * FROM READ_CSV_AUTO('./dataset/googlelocal/business_description.csv');
/* Expansion table (20 rows) */ CREATE TABLE exp (e INT);
INSERT INTO exp SELECT * FROM GENERATE_SERIES(1, 20);
-- query (BlendSQL)
WITH __b1 AS (
  /* Subquery: semantic projection on 79 businesses */
  SELECT
    name AS c__name,
    state AS c__state,
    description AS c__description,
    CONCAT(
      CAST(name AS TEXT),
      ' in ',
      CAST(state AS TEXT),
      '. Description: ',
      CAST(description AS TEXT)
    ) AS __p0
  FROM business_description
), __d1 AS (
  SELECT
    __b1.c__name AS name,
    __b1.c__state AS state,
    __b1.c__description AS description,
    CASE COALESCE({{LLMMap('Classify location: (1) Urban (2) Suburban (3) Rural. Return only number:', __b1.__p0, return_type='int')}}, 0) WHEN 1 THEN 'Urban' WHEN 2 THEN 'Suburban' ELSE 'Rural' END AS location_type
  FROM __b1
)
/* Query: Classify business location type, cross join, limit to 4 */
SELECT
  name,
  e,
  location_type
FROM __d1 AS tagged_businesses
CROSS JOIN exp
LIMIT 4
