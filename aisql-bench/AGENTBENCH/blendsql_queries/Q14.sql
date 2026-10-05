-- setup (DuckDB backend)
/* File: synth_googlelocal_q8.sql */ /* Label: Q14 */ /* Question: Which Southern California campgrounds had 2021 reviews mentioning off-road trails or dirt biking? */ /* Which Southern California campgrounds had 2021 reviews mentioning off-road trails or dirt biking? */ DROP TABLE IF EXISTS business_description;
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
  WHERE
    NOT bd.num_of_reviews IS NULL AND CAST(bd.num_of_reviews AS INT) >= 5
), campgrounds AS (
  SELECT
    __b1.bd__gmap_id AS gmap_id,
    __b1.bd__name AS name,
    __b1.bd__description AS description,
    CAST(__b1.bd__num_of_reviews AS INT) AS num_reviews,
    __b1.bd__state AS state
  FROM __b1
  WHERE
    {{LLMMap('Return true only if this place is a campground, RV park, off-road recreation area, or outdoor trailhead. Otherwise return false.', __b1.__p0)}} = TRUE
), trail_reviews AS (
  SELECT
    c.gmap_id,
    c.name,
    c.num_reviews,
    c.state,
    r.time AS review_time,
    CAST(r.rating AS DOUBLE) AS rating,
    r.text
  FROM campgrounds AS c
  JOIN reviews AS r
    ON r.gmap_id = c.gmap_id
  WHERE
    NOT r.text IS NULL
    AND LENGTH(r.text) >= 80
    AND CAST(r.rating AS DOUBLE) >= 4
    AND r.time ILIKE '%2021%'
    AND (
      LOWER(r.text) LIKE '%dirt bike%'
      OR LOWER(r.text) LIKE '%trail%'
      OR LOWER(r.text) LIKE '%off-road%'
      OR LOWER(r.text) LIKE '%camp%'
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
FROM trail_reviews
ORDER BY
  rating DESC,
  review_time DESC
