-- setup (DuckDB backend)
/* File: synth_yelp_q6.sql */ /* Label: Q23 */ /* Question: Yelp Q6: Top-rated business (Jan-Jun 2016) with category detail and sufficient reviews. */ /* Yelp Q6: Top-rated business (Jan-Jun 2016) with category detail and sufficient reviews. */ DROP TABLE IF EXISTS yelp_business;
CREATE TABLE yelp_business AS SELECT * FROM READ_CSV_AUTO('./dataset/yelp/yelp_business_csv/business.csv', maximum_line_size = 1048576);
DROP TABLE IF EXISTS yelp_review;
CREATE TABLE yelp_review AS SELECT * FROM READ_CSV_AUTO('./dataset/yelp/review.csv', maximum_line_size = 1048576);
DROP TABLE IF EXISTS yelp_tip;
CREATE TABLE yelp_tip AS SELECT * FROM READ_CSV_AUTO('./dataset/yelp/tip.csv', maximum_line_size = 1048576);
-- query (BlendSQL)
WITH __b1 AS (
  SELECT
    yb.business_id AS yb__business_id,
    yb.name AS yb__name,
    yb.attributes AS yb__attributes,
    yb.description AS yb__description,
    CONCAT(
      'Business ID: ',
      CAST(yb.business_id AS TEXT),
      '
Name: ',
      CAST(yb.name AS TEXT),
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
), semantic_category AS (
  SELECT
    __b1.yb__business_id AS business_id,
    __b1.yb__name AS name,
    __b1.yb__attributes AS attributes,
    __b1.yb__description AS description
  FROM __b1
  WHERE
    {{LLMMap('Return true if the description makes the primary business category explicit (for example, Italian restaurant, dental clinic, yoga studio). Favor entries where you can clearly read at least one category phrase.', __b1.__p0)}} = TRUE
), category_text AS (
  SELECT
    sc.business_id,
    sc.name,
    sc.description,
    COALESCE(
      REGEXP_EXTRACT(
        sc.description,
        '(?:services(?: and products)? in|services including|services, including|including|destination for|seeking|specializes in|such as|in the fields of|selection of|range of|array of|variety of|mix of|collection of|menu featuring|options for|dedicated to|catering to)\s+(.+?)(?:\.|$)',
        1
      ),
      ''
    ) AS raw_categories
  FROM semantic_category AS sc
), category_tokens AS (
  SELECT
    ct.business_id,
    ct.name,
    LOWER(TRIM(token.raw_token)) AS category
  FROM category_text AS ct, LATERAL UNNEST(STR_SPLIT(
    REPLACE(REPLACE(REPLACE(ct.raw_categories, ', and ', ', '), ' and ', ', '), '  ', ' '),
    ','
  )) AS token(raw_token)
  WHERE
    TRIM(token.raw_token) <> ''
), reviews_link AS (
  SELECT
    'businessid_' || REGEXP_EXTRACT(r.business_ref, '([0-9]+)$') AS business_id,
    r.rating,
    COALESCE(
      TRY_STRPTIME(r.date, '%Y-%m-%d %H:%M:%S'),
      TRY_STRPTIME(r.date, '%d %b %Y, %H:%M'),
      TRY_STRPTIME(r.date, '%d %b %Y %H:%M'),
      TRY_STRPTIME(r.date, '%d %b %Y, %I:%M %p'),
      TRY_STRPTIME(r.date, '%d %B %Y, %H:%M'),
      TRY_STRPTIME(r.date, '%d %B %Y, %I:%M %p'),
      TRY_STRPTIME(r.date, '%B %d, %Y at %I:%M %p'),
      TRY_STRPTIME(r.date, '%B %d, %Y %H:%M %p'),
      TRY_STRPTIME(r.date, '%d %B %Y at %I:%M %p'),
      TRY_STRPTIME(r.date, '%d %b %Y at %I:%M %p')
    ) AS review_ts
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
/* Non-aggregate: return individual reviews (Jan-Jun 2016) with category and rating */
SELECT
  sc.name AS business_name,
  ct.category,
  rl.review_ts AS review_time,
  rl.rating
FROM semantic_category AS sc
JOIN reviews_link AS rl
  ON rl.business_id = sc.business_id
JOIN category_tokens AS ct
  ON ct.business_id = sc.business_id
WHERE
  rl.review_ts BETWEEN CAST('2016-01-01 00:00:00' AS TIMESTAMP) AND CAST('2016-06-30 23:59:59' AS TIMESTAMP)
  AND NOT rl.review_ts IS NULL
ORDER BY
  rl.rating DESC,
  rl.review_ts DESC
