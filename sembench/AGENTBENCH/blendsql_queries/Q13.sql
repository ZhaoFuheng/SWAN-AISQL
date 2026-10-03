-- setup (DuckDB backend)
/* File: synth_googlelocal_q7.sql */ /* Label: Q13 */ /* Question: Which open California nail salons received detailed 2021 reviews highlighting manicure or brow results? */ /* Which open California nail salons received detailed 2021 reviews highlighting manicure or brow results? */ DROP TABLE IF EXISTS business_description;
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
    NOT bd.num_of_reviews IS NULL AND CAST(bd.num_of_reviews AS INT) >= 3
), nail_salons AS (
  SELECT
    __b1.bd__gmap_id AS gmap_id,
    __b1.bd__name AS name,
    __b1.bd__description AS description,
    CAST(__b1.bd__num_of_reviews AS INT) AS num_reviews,
    __b1.bd__state AS state
  FROM __b1
  WHERE
    {{LLMMap('Return true only if this business primarily offers nail salon, manicure, pedicure, brow, or spa services. Otherwise return false.', __b1.__p0)}} = TRUE
), beauty_feedback AS (
  SELECT
    n.gmap_id,
    n.name,
    n.num_reviews,
    n.state,
    r.time AS review_time,
    CAST(r.rating AS DOUBLE) AS rating,
    r.text
  FROM nail_salons AS n
  JOIN reviews AS r
    ON r.gmap_id = n.gmap_id
  WHERE
    NOT r.text IS NULL
    AND LENGTH(r.text) >= 120
    AND CAST(r.rating AS DOUBLE) >= 4.5
    AND r.time ILIKE '%2021%'
    AND (
      LOWER(r.text) LIKE '%nail%'
      OR LOWER(r.text) LIKE '%mani%'
      OR LOWER(r.text) LIKE '%pedi%'
      OR LOWER(r.text) LIKE '%brow%'
      OR LOWER(r.text) LIKE '%dip%'
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
FROM beauty_feedback
ORDER BY
  rating DESC,
  review_time DESC
