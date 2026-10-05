-- setup (DuckDB backend)
/* File: synth_googlelocal_q6.sql */ /* Label: Q12 */ /* Question: Which open California auto repair shops earned long 2021 reviews praising honest or fast service? */ /* Which open California auto repair shops earned long 2021 reviews praising honest or fast service? */ DROP TABLE IF EXISTS business_description;
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
    NOT bd.num_of_reviews IS NULL
    AND CAST(bd.num_of_reviews AS INT) >= 10
    AND bd.state ILIKE 'Open%'
), auto_shops AS (
  SELECT
    __b1.bd__gmap_id AS gmap_id,
    __b1.bd__name AS name,
    __b1.bd__description AS description,
    CAST(__b1.bd__num_of_reviews AS INT) AS num_reviews,
    __b1.bd__state AS state
  FROM __b1
  WHERE
    {{LLMMap('Return true only if this business is primarily an auto repair, mechanic, brake, tire, or vehicle maintenance shop. Otherwise return false.', __b1.__p0)}} = TRUE
), honest_reviews AS (
  SELECT
    s.gmap_id,
    s.name,
    s.num_reviews,
    s.state,
    r.time AS review_time,
    CAST(r.rating AS DOUBLE) AS rating,
    r.text,
    LENGTH(r.text) AS text_length
  FROM auto_shops AS s
  JOIN reviews AS r
    ON r.gmap_id = s.gmap_id
  WHERE
    NOT r.text IS NULL
    AND LENGTH(r.text) >= 120
    AND CAST(r.rating AS DOUBLE) >= 4.5
    AND r.time ILIKE '%2021%'
    AND (
      LOWER(r.text) LIKE '%honest%'
      OR LOWER(r.text) LIKE '%quick%'
      OR LOWER(r.text) LIKE '%fast%'
    )
)
SELECT
  gmap_id,
  name,
  num_reviews,
  state,
  review_time,
  rating,
  text,
  text_length
FROM honest_reviews
ORDER BY
  rating DESC,
  text_length DESC
