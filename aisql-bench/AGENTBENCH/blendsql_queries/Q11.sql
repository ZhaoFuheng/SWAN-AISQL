-- setup (DuckDB backend)
/* File: synth_googlelocal_q5.sql */ /* Label: Q11 */ /* Question: Which California cannabis dispensaries received long, high-rated 2021 reviews praising helpful or friendly staff? */ /* Which California cannabis dispensaries received long, high-rated 2021 reviews praising helpful or friendly staff? */ DROP TABLE IF EXISTS business_description;
DROP TABLE IF EXISTS reviews;
CREATE TABLE business_description AS SELECT * FROM READ_CSV_AUTO('./dataset/googlelocal/business_description.csv');
CREATE TABLE reviews AS SELECT * FROM READ_CSV_AUTO('./dataset/googlelocal/review.csv');
-- query (BlendSQL)
WITH __b1 AS (
  SELECT
    bd.gmap_id AS bd__gmap_id,
    bd.name AS bd__name,
    bd.description AS bd__description,
    bd.state AS bd__state,
    bd.num_of_reviews AS bd__num_of_reviews,
    CONCAT(
      'Name: ',
      CAST(bd.name AS TEXT),
      '
Description: ',
      CAST(bd.description AS TEXT),
      '
Misc: ',
      CAST(bd.misc AS TEXT),
      '
Respond with true or false only.'
    ) AS __p0
  FROM business_description AS bd
), dispensaries AS (
  SELECT
    __b1.bd__gmap_id AS gmap_id,
    __b1.bd__name AS name,
    __b1.bd__description AS description,
    CAST(__b1.bd__num_of_reviews AS INT) AS num_reviews,
    __b1.bd__state AS state
  FROM __b1
  WHERE
    {{LLMMap('Return true only if this business is primarily a cannabis dispensary, delivery service, or weed retailer. Otherwise return false.', __b1.__p0)}} = TRUE
), staff_praise AS (
  SELECT
    d.gmap_id,
    d.name,
    d.num_reviews,
    d.state,
    r.time AS review_time,
    CAST(r.rating AS DOUBLE) AS rating,
    r.text
  FROM dispensaries AS d
  JOIN reviews AS r
    ON r.gmap_id = d.gmap_id
  WHERE
    NOT r.text IS NULL
    AND LENGTH(r.text) >= 160
    AND CAST(r.rating AS DOUBLE) >= 4.8
    AND r.time ILIKE '%2021%'
    AND LOWER(r.text) LIKE '%staff%'
    AND (
      LOWER(r.text) LIKE '%helpful%'
      OR LOWER(r.text) LIKE '%friendly%'
      OR LOWER(r.text) LIKE '%knowledgeable%'
      OR LOWER(r.text) LIKE '%budtender%'
      OR LOWER(r.text) LIKE '%bud tender%'
    )
)
SELECT
  gmap_id,
  name,
  num_reviews,
  state,
  review_time,
  rating,
  text
FROM staff_praise
ORDER BY
  rating DESC,
  review_time DESC,
  LENGTH(text) DESC
