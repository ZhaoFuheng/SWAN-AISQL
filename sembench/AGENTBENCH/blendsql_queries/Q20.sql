-- setup (DuckDB backend)
/* File: synth_yelp_q3.sql */ /* Label: Q20 */ /* Question: Yelp Q3: Businesses with 2018 reviews that offer customer or bike parking. */ /* Yelp Q3: Businesses with 2018 reviews that offer customer or bike parking. */ DROP TABLE IF EXISTS yelp_business;
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
Respond with true or false only.'
    ) AS __p0
  FROM yelp_business AS yb
), semantic_parking AS (
  SELECT
    __b1.yb__business_id AS business_id,
    __b1.yb__attributes AS attributes,
    __b1.yb__description AS description
  FROM __b1
  WHERE
    {{LLMMap('Return true if the business offers customer parking (lot, garage, street, or valet) or any form of bike parking. Use the provided attributes and description to decide.', __b1.__p0)}} = TRUE
), reviews_link AS (
  SELECT
    'businessid_' || REGEXP_EXTRACT(r.business_ref, '([0-9]+)$') AS business_id,
    r.date AS review_date,
    r.rating
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
  sp.business_id,
  rl.review_date,
  rl.rating
FROM semantic_parking AS sp
JOIN reviews_link AS rl
  ON rl.business_id = sp.business_id
LEFT JOIN tips_link AS tl
  ON tl.business_id = sp.business_id
WHERE
  rl.review_date ILIKE '%2018%'
  AND rl.rating >= 4
  AND sp.attributes ILIKE '%"BikeParking": "True"%'
  AND tl.tip_date ILIKE '%2018%'
  AND sp.description ILIKE '%Philadelphia%'
ORDER BY
  rl.rating DESC,
  rl.review_date DESC
