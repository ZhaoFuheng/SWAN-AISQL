-- setup (DuckDB backend)
/* File: synth_yelp_q4.sql */ /* Label: Q21 */ /* Question: Yelp Q4: Category with most credit-card-friendly businesses and their average rating. */ /* Yelp Q4: Category with most credit-card-friendly businesses and their average rating. */ DROP TABLE IF EXISTS yelp_business;
CREATE TABLE yelp_business AS SELECT * FROM READ_CSV_AUTO('./dataset/yelp/yelp_business_csv/business.csv', maximum_line_size = 1048576);
DROP TABLE IF EXISTS yelp_review;
CREATE TABLE yelp_review AS SELECT * FROM READ_CSV_AUTO('./dataset/yelp/review.csv', maximum_line_size = 1048576);
DROP TABLE IF EXISTS yelp_tip;
CREATE TABLE yelp_tip AS SELECT * FROM READ_CSV_AUTO('./dataset/yelp/tip.csv', maximum_line_size = 1048576);
-- query (BlendSQL)
WITH __b1 AS (
  SELECT
    yb.business_id AS yb__business_id,
    yb.attributes AS yb__attributes,
    yb.description AS yb__description,
    CONCAT(
      'Business ID: ',
      CAST(yb.business_id AS TEXT),
      '
Attributes: ',
      CAST(yb.attributes AS TEXT),
      '
Description: ',
      CAST(yb.description AS TEXT),
      '
Answer true or false only.'
    ) AS __p0
  FROM yelp_business AS yb
), semantic_credit AS (
  SELECT
    __b1.yb__business_id AS business_id,
    __b1.yb__attributes AS attributes,
    __b1.yb__description AS description
  FROM __b1
  WHERE
    {{LLMMap('Return true if the business clearly accepts credit card payments. Prefer explicit evidence from attributes or the narrative.', __b1.__p0)}} = TRUE
), normalized_categories AS (
  SELECT
    sc.business_id,
    REGEXP_REPLACE(
      COALESCE(
        REGEXP_EXTRACT(
          sc.description,
          '(?:services(?: and products)? in|services including|services, including|including|destination for|seeking|specializes in|such as|in the fields of|selection of|range of|array of|variety of|mix of|collection of|menu featuring|options for|dedicated to|catering to)\s+(.+?)(?:\.|$)',
          1
        ),
        ''
      ),
      '^(?:services(?: and products)? in|services including|services, including|including|destination for|seeking|specializes in|such as|in the fields of|selection of|range of|array of|variety of|mix of|collection of|menu featuring|options for|dedicated to|catering to)\s+',
      ''
    ) AS raw_categories
  FROM semantic_credit AS sc
), category_tokens AS (
  SELECT
    sc.business_id,
    LOWER(TRIM(tokens.raw_token)) AS category
  FROM normalized_categories AS sc, LATERAL UNNEST(STR_SPLIT(
    REPLACE(REPLACE(REPLACE(sc.raw_categories, ', and ', ', '), ' and ', ', '), '  ', ' '),
    ','
  )) AS tokens(raw_token)
  WHERE
    TRIM(tokens.raw_token) <> ''
), reviews_link AS (
  SELECT
    'businessid_' || REGEXP_EXTRACT(r.business_ref, '([0-9]+)$') AS business_id,
    r.rating,
    r.date AS review_date
  FROM yelp_review AS r
  WHERE
    r.business_ref LIKE 'businessref_%'
), tips_link AS (
  SELECT
    'businessid_' || REGEXP_EXTRACT(t.business_ref, '([0-9]+)$') AS business_id,
    t.date AS tip_date
  FROM yelp_tip AS t
  WHERE
    t.business_ref LIKE 'businessref_%'
)
SELECT
  ct.category,
  sc.business_id,
  rl.review_date,
  rl.rating
FROM category_tokens AS ct
JOIN semantic_credit AS sc
  ON sc.business_id = ct.business_id
JOIN reviews_link AS rl
  ON rl.business_id = ct.business_id
JOIN tips_link AS tl
  ON tl.business_id = ct.business_id
WHERE
  sc.attributes ILIKE '%"BusinessAcceptsCreditCards": "True"%'
  AND rl.review_date ILIKE '%2019%'
  AND tl.tip_date ILIKE '%2019%'
ORDER BY
  rl.rating DESC,
  rl.review_date DESC
