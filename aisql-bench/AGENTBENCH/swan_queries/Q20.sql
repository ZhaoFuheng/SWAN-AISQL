-- File: synth_yelp_q3.sql
-- Label: Q20
-- Question: Yelp Q3: Businesses with 2018 reviews that offer customer or bike parking.

-- Yelp Q3: Businesses with 2018 reviews that offer customer or bike parking.
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

WITH semantic_parking AS (
	SELECT
		yb.business_id,
		yb.attributes,
		yb.description
	FROM yelp_business yb
	WHERE ai_filter('Return true if the business offers customer parking (lot, garage, street, or valet) or any form of bike parking. Use the provided attributes and description to decide.
Business ID: ' || yb.business_id || '
Attributes: ' || yb.attributes || '
Description: ' || yb.description || '
Respond with true or false only.' || e'\n Return a single yes or no, do not contain any other words.')
),
reviews_link AS (
	SELECT
		'businessid_' || regexp_extract(r.business_ref, '([0-9]+)$') AS business_id,
		r.date AS review_date,
		r.rating
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
	sp.business_id,
	rl.review_date,
	rl.rating
FROM semantic_parking sp
JOIN reviews_link rl ON rl.business_id = sp.business_id
LEFT JOIN tips_link tl ON tl.business_id = sp.business_id
WHERE rl.review_date ILIKE '%2018%'
	AND rl.rating >= 4
	AND sp.attributes ILIKE '%"BikeParking": "True"%'
	AND tl.tip_date ILIKE '%2018%'
	AND sp.description ILIKE '%Philadelphia%'
ORDER BY rl.rating DESC, rl.review_date DESC;
