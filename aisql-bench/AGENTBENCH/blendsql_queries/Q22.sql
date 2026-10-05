-- setup (DuckDB backend)
/* File: synth_yelp_q5.sql */ /* Label: Q22 */ /* Question: Yelp Q5: U.S. state with the most WiFi-enabled businesses and their average rating. */ /* Yelp Q5: U.S. state with the most WiFi-enabled businesses and their average rating. */ DROP TABLE IF EXISTS yelp_business;
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
Respond true or false only.'
    ) AS __p0
  FROM yelp_business AS yb
), semantic_wifi AS (
  SELECT
    __b1.yb__business_id AS business_id,
    __b1.yb__attributes AS attributes,
    __b1.yb__description AS description
  FROM __b1
  WHERE
    {{LLMMap('Return true if the business offers public WiFi access for its customers. Use both the attributes and description for evidence.', __b1.__p0)}} = TRUE
), state_lookup AS (
  SELECT
    sw.business_id,
    REGEXP_EXTRACT(sw.description, ',\s*([A-Z]{2})\b', 1) AS state_code
  FROM semantic_wifi AS sw
), valid_states AS (
  SELECT
    business_id,
    state_code
  FROM state_lookup
  WHERE
    state_code IN (
      'AL',
      'AK',
      'AZ',
      'AR',
      'CA',
      'CO',
      'CT',
      'DE',
      'FL',
      'GA',
      'HI',
      'ID',
      'IL',
      'IN',
      'IA',
      'KS',
      'KY',
      'LA',
      'ME',
      'MD',
      'MA',
      'MI',
      'MN',
      'MS',
      'MO',
      'MT',
      'NE',
      'NV',
      'NH',
      'NJ',
      'NM',
      'NY',
      'NC',
      'ND',
      'OH',
      'OK',
      'OR',
      'PA',
      'RI',
      'SC',
      'SD',
      'TN',
      'TX',
      'UT',
      'VT',
      'VA',
      'WA',
      'WV',
      'WI',
      'WY'
    )
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
  vs.state_code AS state,
  sw.business_id,
  rl.review_date,
  rl.rating
FROM valid_states AS vs
JOIN semantic_wifi AS sw
  ON sw.business_id = vs.business_id
JOIN reviews_link AS rl
  ON rl.business_id = vs.business_id
JOIN tips_link AS tl
  ON tl.business_id = vs.business_id
WHERE
  sw.attributes ILIKE '%"WiFi":%'
  AND rl.review_date ILIKE '%2018%'
  AND tl.tip_date ILIKE '%2018%'
ORDER BY
  rl.rating DESC,
  rl.review_date DESC
