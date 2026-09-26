-- File: synth_yelp_q6.sql
-- Label: Q23
-- Question: Yelp Q6: Top-rated business (Jan-Jun 2016) with category detail and sufficient reviews.

-- Yelp Q6: Top-rated business (Jan-Jun 2016) with category detail and sufficient reviews.
DROP TABLE IF EXISTS yelp_business;
CREATE TABLE yelp_business AS
SELECT *
FROM read_csv_auto('./dataset/yelp/yelp_business_csv/business.csv', maximum_line_size=1048576);

DROP TABLE IF EXISTS yelp_review;
CREATE TABLE yelp_review AS
SELECT *
FROM read_csv_auto('./dataset/yelp/review.csv', maximum_line_size=1048576);

DROP TABLE IF EXISTS yelp_tip;
CREATE TABLE yelp_tip AS
SELECT *
FROM read_csv_auto('./dataset/yelp/tip.csv', maximum_line_size=1048576);

WITH semantic_category AS (
	SELECT
		yb.business_id,
		yb.name,
		yb.attributes,
		yb.description
	FROM yelp_business yb
	WHERE (lower(ai_complete('Return true if the description makes the primary business category explicit (for example, Italian restaurant, dental clinic, yoga studio). Favor entries where you can clearly read at least one category phrase.
Business ID: ' || yb.business_id || '
Name: ' || yb.name || '
Attributes: ' || yb.attributes || '
Description: ' || yb.description || '
Answer true or false only.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
),
category_text AS (
	SELECT
		sc.business_id,
		sc.name,
		sc.description,
		coalesce(
			regexp_extract(sc.description, '(?:services(?: and products)? in|services including|services, including|including|destination for|seeking|specializes in|such as|in the fields of|selection of|range of|array of|variety of|mix of|collection of|menu featuring|options for|dedicated to|catering to)\s+(.+?)(?:\.|$)', 1),
			''
		) AS raw_categories
	FROM semantic_category sc
),
category_tokens AS (
	SELECT
		ct.business_id,
		ct.name,
		lower(trim(token.raw_token)) AS category
	FROM category_text ct,
	LATERAL UNNEST(
		string_split(
			replace(replace(replace(ct.raw_categories, ', and ', ', '), ' and ', ', '), '  ', ' '),
			','
		)
	) AS token(raw_token)
	WHERE trim(token.raw_token) <> ''
),
reviews_link AS (
	SELECT
		'businessid_' || regexp_extract(r.business_ref, '([0-9]+)$') AS business_id,
		r.rating,
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
tips_link AS (
	SELECT
		'businessid_' || regexp_extract(t.business_ref, '([0-9]+)$') AS business_id,
		t.date AS tip_date
	FROM yelp_tip t
	WHERE t.business_ref LIKE 'businessref_%'
)
-- Non-aggregate: return individual reviews (Jan-Jun 2016) with category and rating
SELECT
  sc.name AS business_name,
  ct.category,
  rl.review_ts AS review_time,
  rl.rating
FROM semantic_category sc
JOIN reviews_link rl ON rl.business_id = sc.business_id
JOIN category_tokens ct ON ct.business_id = sc.business_id
WHERE rl.review_ts BETWEEN TIMESTAMP '2016-01-01 00:00:00' AND TIMESTAMP '2016-06-30 23:59:59'
  AND rl.review_ts IS NOT NULL
ORDER BY rl.rating DESC, rl.review_ts DESC;
