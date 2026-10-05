-- File: synth_yelp_q4.sql
-- Label: Q21
-- Question: Yelp Q4: Category with most credit-card-friendly businesses and their average rating.

-- Yelp Q4: Category with most credit-card-friendly businesses and their average rating.
DROP TABLE IF EXISTS yelp_business;
CREATE TABLE yelp_business AS
SELECT *
FROM read_csv_auto('./test/semantic/agent_bench/dataset/yelp/yelp_business_csv/business.csv', maximum_line_size=1048576);

DROP TABLE IF EXISTS yelp_review;
CREATE TABLE yelp_review AS
SELECT *
FROM read_csv_auto('./test/semantic/agent_bench/dataset/yelp/review.csv', maximum_line_size=1048576);

DROP TABLE IF EXISTS yelp_tip;
CREATE TABLE yelp_tip AS
SELECT *
FROM read_csv_auto('./test/semantic/agent_bench/dataset/yelp/tip.csv', maximum_line_size=1048576);

WITH semantic_credit AS (
	SELECT
		yb.business_id,
		yb.attributes,
		yb.description
	FROM yelp_business yb
	WHERE SEMANTIC('Return true if the business clearly accepts credit card payments. Prefer explicit evidence from attributes or the narrative.
Business ID: {yb.business_id}
Attributes: {yb.attributes}
Description: {yb.description}
Answer true or false only.')
),
normalized_categories AS (
	SELECT
		sc.business_id,
		regexp_replace(
			coalesce(
				regexp_extract(sc.description, '(?:services(?: and products)? in|services including|services, including|including|destination for|seeking|specializes in|such as|in the fields of|selection of|range of|array of|variety of|mix of|collection of|menu featuring|options for|dedicated to|catering to)\s+(.+?)(?:\.|$)', 1),
				''
			),
			'^(?:services(?: and products)? in|services including|services, including|including|destination for|seeking|specializes in|such as|in the fields of|selection of|range of|array of|variety of|mix of|collection of|menu featuring|options for|dedicated to|catering to)\s+',
			''
		) AS raw_categories
	FROM semantic_credit sc
),
category_tokens AS (
	SELECT
		sc.business_id,
		lower(trim(tokens.raw_token)) AS category
	FROM normalized_categories sc,
	LATERAL UNNEST(
		string_split(
			replace(replace(replace(sc.raw_categories, ', and ', ', '), ' and ', ', '), '  ', ' '),
			','
		)
	) AS tokens(raw_token)
	WHERE trim(tokens.raw_token) <> ''
),
reviews_link AS (
	SELECT
		'businessid_' || regexp_extract(r.business_ref, '([0-9]+)$') AS business_id,
		r.rating,
		r.date AS review_date
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
SELECT
	ct.category,
	sc.business_id,
	rl.review_date,
	rl.rating
FROM category_tokens ct
JOIN semantic_credit sc ON sc.business_id = ct.business_id
JOIN reviews_link rl ON rl.business_id = ct.business_id
JOIN tips_link tl ON tl.business_id = ct.business_id
WHERE sc.attributes ILIKE '%"BusinessAcceptsCreditCards": "True"%'
	AND rl.review_date ILIKE '%2019%'
	AND tl.tip_date ILIKE '%2019%'
ORDER BY rl.rating DESC, rl.review_date DESC;
