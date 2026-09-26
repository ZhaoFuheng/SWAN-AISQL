-- File: synth_googlelocal_q5.sql
-- Label: Q11
-- Question: Which California cannabis dispensaries received long, high-rated 2021 reviews praising helpful or friendly staff?

-- Which California cannabis dispensaries received long, high-rated 2021 reviews praising helpful or friendly staff?
DROP TABLE IF EXISTS business_description;
DROP TABLE IF EXISTS reviews;

CREATE TABLE business_description AS
SELECT *
FROM read_csv_auto('./dataset/googlelocal/business_description.csv');

CREATE TABLE reviews AS
SELECT *
FROM read_csv_auto('./dataset/googlelocal/review.csv');

WITH
dispensaries AS (
	SELECT
		bd.gmap_id,
		bd.name,
		bd.description,
		CAST(bd.num_of_reviews AS INTEGER) AS num_reviews,
		bd.state
	FROM business_description bd
	WHERE (lower(ai_complete('Return true only if this business is primarily a cannabis dispensary, delivery service, or weed retailer. Otherwise return false.
Name: ' || bd.name || '
Description: ' || bd.description || '
Misc: ' || bd.misc || '
Respond with true or false only.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
),
staff_praise AS (
	SELECT
		d.gmap_id,
		d.name,
		d.num_reviews,
		d.state,
		r.time AS review_time,
		CAST(r.rating AS DOUBLE) AS rating,
		r.text
	FROM dispensaries d
	JOIN reviews r ON r.gmap_id = d.gmap_id
	WHERE r.text IS NOT NULL
		AND LENGTH(r.text) >= 160
		AND CAST(r.rating AS DOUBLE) >= 4.8
		AND r.time ILIKE '%2021%'
		AND LOWER(r.text) LIKE '%staff%'
		AND (
				LOWER(r.text) LIKE '%helpful%'
			OR LOWER(r.text) LIKE '%friendly%'
			OR LOWER(r.text) LIKE '%knowledgeable%'
			OR LOWER(r.text) LIKE '%budtender%'
			OR LOWER(r.text) LIKE '%bud tender%'
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
FROM staff_praise
ORDER BY rating DESC, review_time DESC, LENGTH(text) DESC;

