-- setup (DuckDB backend)
/* File: synth_yelp_q7.sql */ /* Label: Q24 */ /* Question: Yelp Q7: Top five categories by review volume from users who joined in 2016. */ /* Yelp Q7: Top five categories by review volume from users who joined in 2016. */ DROP TABLE IF EXISTS yelp_business;
CREATE TABLE yelp_business AS SELECT * FROM READ_CSV_AUTO('./dataset/yelp/yelp_business_csv/business.csv', maximum_line_size = 1048576);
DROP TABLE IF EXISTS yelp_review;
CREATE TABLE yelp_review AS SELECT * FROM READ_CSV_AUTO('./dataset/yelp/review.csv', maximum_line_size = 1048576);
DROP TABLE IF EXISTS yelp_user;
CREATE TABLE yelp_user AS SELECT * FROM READ_CSV_AUTO('./dataset/yelp/user.csv', maximum_line_size = 1048576);
-- query (BlendSQL)
WITH user_joined_2016 AS (
  SELECT
    u.user_id,
    COALESCE(
      TRY_STRPTIME(u.yelping_since, '%Y-%m-%d %H:%M:%S'),
      TRY_STRPTIME(u.yelping_since, '%d %b %Y, %H:%M'),
      TRY_STRPTIME(u.yelping_since, '%d %B %Y, %H:%M'),
      TRY_STRPTIME(u.yelping_since, '%B %d, %Y at %I:%M %p'),
      TRY_STRPTIME(u.yelping_since, '%d %b %Y at %I:%M %p'),
      TRY_STRPTIME(u.yelping_since, '%d %B %Y at %I:%M %p')
    ) AS joined_ts
  FROM yelp_user AS u
), qualified_users AS (
  SELECT
    user_id
  FROM user_joined_2016
  WHERE
    joined_ts BETWEEN CAST('2016-01-01 00:00:00' AS TIMESTAMP) AND CAST('2016-12-31 23:59:59' AS TIMESTAMP)
), parsed_reviews AS (
  SELECT
    'businessid_' || REGEXP_EXTRACT(r.business_ref, '([0-9]+)$') AS business_id,
    r.user_id,
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
), filtered_reviews AS (
  SELECT
    pr.business_id,
    pr.user_id,
    pr.review_ts
  FROM parsed_reviews AS pr
  JOIN qualified_users AS qu
    ON qu.user_id = pr.user_id
  WHERE
    pr.review_ts >= CAST('2016-01-01 00:00:00' AS TIMESTAMP)
    AND NOT pr.review_ts IS NULL
), active_businesses AS (
  SELECT DISTINCT
    business_id
  FROM filtered_reviews
), __b1 AS (
  SELECT
    yb.business_id AS yb__business_id,
    yb.attributes AS yb__attributes,
    yb.description AS yb__description,
    ab.business_id AS ab__business_id,
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
  JOIN active_businesses AS ab
    ON ab.business_id = yb.business_id
), semantic_category AS (
  SELECT
    __b1.yb__business_id AS business_id,
    __b1.yb__attributes AS attributes,
    __b1.yb__description AS description
  FROM __b1
  WHERE
    {{LLMMap('Return true if the business description clearly lists multiple categories or service types that can be tokenized. Focus on entries with explicit category phrases.', __b1.__p0)}} = TRUE
), category_text AS (
  SELECT
    sc.business_id,
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
    LOWER(TRIM(token.raw_token)) AS category
  FROM category_text AS ct, LATERAL UNNEST(STR_SPLIT(
    REPLACE(REPLACE(REPLACE(ct.raw_categories, ', and ', ', '), ' and ', ', '), '  ', ' '),
    ','
  )) AS token(raw_token)
  WHERE
    TRIM(token.raw_token) <> ''
)
/* Non-aggregate: return recent reviews by users who joined in 2016 with extracted categories */
SELECT
  ct.category,
  fr.user_id,
  fr.review_ts
FROM category_tokens AS ct
JOIN filtered_reviews AS fr
  ON fr.business_id = ct.business_id
ORDER BY
  fr.review_ts DESC
