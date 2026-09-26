-- File: synth_yelp_q7.sql
-- Label: Q24
-- Question: Yelp Q7: Top five categories by review volume from users who joined in 2016.

-- Yelp Q7: Top five categories by review volume from users who joined in 2016.
DROP TABLE IF EXISTS yelp_business;
CREATE TABLE yelp_business AS
SELECT *
FROM read_csv_auto('./dataset/yelp/yelp_business_csv/business.csv', maximum_line_size=1048576);

DROP TABLE IF EXISTS yelp_review;
CREATE TABLE yelp_review AS
SELECT *
FROM read_csv_auto('./dataset/yelp/review.csv', maximum_line_size=1048576);

DROP TABLE IF EXISTS yelp_user;
CREATE TABLE yelp_user AS
SELECT *
FROM read_csv_auto('./dataset/yelp/user.csv', maximum_line_size=1048576);

WITH user_joined_2016 AS (
	SELECT
		u.user_id,
		COALESCE(
			try_strptime(u.yelping_since, '%Y-%m-%d %H:%M:%S'),
			try_strptime(u.yelping_since, '%d %b %Y, %H:%M'),
			try_strptime(u.yelping_since, '%d %B %Y, %H:%M'),
			try_strptime(u.yelping_since, '%B %d, %Y at %I:%M %p'),
			try_strptime(u.yelping_since, '%d %b %Y at %I:%M %p'),
			try_strptime(u.yelping_since, '%d %B %Y at %I:%M %p')
		) AS joined_ts
	FROM yelp_user u
),
qualified_users AS (
	SELECT user_id
	FROM user_joined_2016
	WHERE joined_ts BETWEEN TIMESTAMP '2016-01-01 00:00:00' AND TIMESTAMP '2016-12-31 23:59:59'
),
parsed_reviews AS (
	SELECT
		'businessid_' || regexp_extract(r.business_ref, '([0-9]+)$') AS business_id,
		r.user_id,
		COALESCE(
			try_strptime(r.date, '%Y-%m-%d %H:%M:%S'),
			try_strptime(r.date, '%d %b %Y, %H:%M'),
			try_strptime(r.date, '%d %b %Y %H:%M'),
			try_strptime(r.date, '%d %b %Y, %I:%M %p'),
			try_strptime(r.date, '%d %B %Y, %H:%M'),
			try_strptime(r.date, '%d %B %Y, %I:%M %p'),
			try_strptime(r.date, '%B %d, %Y at %I:%M %p'),
			try_strptime(r.date, '%B %d, %Y %H:%M %p'),
			try_strptime(r.date, '%d %B %Y at %I:%M %p'),
			try_strptime(r.date, '%d %b %Y at %I:%M %p')
		) AS review_ts
	FROM yelp_review r
	WHERE r.business_ref LIKE 'businessref_%'
),
filtered_reviews AS (
	SELECT
		pr.business_id,
		pr.user_id,
		pr.review_ts
	FROM parsed_reviews pr
	JOIN qualified_users qu ON qu.user_id = pr.user_id
	WHERE pr.review_ts >= TIMESTAMP '2016-01-01 00:00:00'
		AND pr.review_ts IS NOT NULL
),
active_businesses AS (
	SELECT DISTINCT business_id
	FROM filtered_reviews
),
semantic_category AS (
	SELECT
		yb.business_id,
		yb.attributes,
		yb.description
	FROM yelp_business yb
	JOIN active_businesses ab ON ab.business_id = yb.business_id
	WHERE (lower(ai_complete('Return true if the business description clearly lists multiple categories or service types that can be tokenized. Focus on entries with explicit category phrases.
Business ID: ' || yb.business_id || '
Attributes: ' || yb.attributes || '
Description: ' || yb.description || '
Respond with true or false only.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
),
category_text AS (
	SELECT
		sc.business_id,
		coalesce(
			regexp_extract(sc.description, '(?:services(?: and products)? in|services including|services, including|including|destination for|seeking|specializes in|such as|in the fields of|selection of|range of|array of|variety of|mix of|collection of|menu featuring|options for|dedicated to|catering to)\s+(.+?)(?:\.|$)', 1),
			''
		) AS raw_categories
	FROM semantic_category sc
),
category_tokens AS (
	SELECT
		ct.business_id,
		lower(trim(token.raw_token)) AS category
	FROM category_text ct,
	LATERAL UNNEST(
		string_split(
			replace(replace(replace(ct.raw_categories, ', and ', ', '), ' and ', ', '), '  ', ' '),
			','
		)
	) AS token(raw_token)
	WHERE trim(token.raw_token) <> ''
)
-- Non-aggregate: return recent reviews by users who joined in 2016 with extracted categories
SELECT
  ct.category,
  fr.user_id,
  fr.review_ts
FROM category_tokens ct
JOIN filtered_reviews fr ON fr.business_id = ct.business_id
ORDER BY fr.review_ts DESC;
