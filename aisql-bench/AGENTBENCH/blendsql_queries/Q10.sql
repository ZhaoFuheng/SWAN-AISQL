-- setup (DuckDB backend)
/* File: synth_googlelocal_q4.sql */ /* Label: Q10 */ /* Question: Which GoogleLocal massage therapy businesses receive detailed high-rated reviews? */ /* Which GoogleLocal massage therapy businesses receive detailed high-rated reviews? */ DROP TABLE IF EXISTS business_description;
DROP TABLE IF EXISTS reviews;
CREATE TABLE business_description AS SELECT * FROM READ_CSV_AUTO('./dataset/googlelocal/business_description.csv');
CREATE TABLE reviews AS SELECT * FROM READ_CSV_AUTO('./dataset/googlelocal/review.csv');
-- query (BlendSQL)
WITH __b1 AS (
  SELECT
    bd.gmap_id AS bd__gmap_id,
    bd.name AS bd__name,
    bd.description AS bd__description,
    bd.num_of_reviews AS bd__num_of_reviews,
    bd.state AS bd__state,
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
Number of Reviews: ',
      CAST(bd.num_of_reviews AS TEXT),
      '
Respond with true or false only.'
    ) AS __p0
  FROM business_description AS bd
  WHERE
    NOT bd.num_of_reviews IS NULL
    AND CAST(bd.num_of_reviews AS INT) >= 15
    AND bd.state LIKE 'Open%'
), massage_businesses AS (
  SELECT
    __b1.bd__gmap_id AS gmap_id,
    __b1.bd__name AS name,
    __b1.bd__description AS description,
    __b1.bd__num_of_reviews AS num_of_reviews,
    __b1.bd__state AS state
  FROM __b1
  WHERE
    {{LLMMap('Return true if this business primarily offers massage therapy or spa treatments. Otherwise return false.', __b1.__p0)}} = TRUE
), positive_reviews AS (
  SELECT
    mb.gmap_id,
    mb.name,
    CAST(mb.num_of_reviews AS INT) AS business_review_count,
    mb.state,
    r.time AS review_time,
    CAST(r.rating AS DOUBLE) AS rating,
    r.text
  FROM massage_businesses AS mb
  JOIN reviews AS r
    ON r.gmap_id = mb.gmap_id
  WHERE
    CAST(r.rating AS DOUBLE) >= 4.8
    AND NOT r.text IS NULL
    AND LENGTH(r.text) >= 150
    AND (
      STRPOS(r.time, '2021') > 0
      OR STRPOS(r.time, '2022') > 0
      OR STRPOS(r.time, '2023') > 0
      OR STRPOS(r.time, '2024') > 0
    )
)
SELECT
  gmap_id,
  name,
  business_review_count,
  state,
  review_time,
  rating,
  text
FROM positive_reviews
ORDER BY
  rating DESC,
  review_time DESC
