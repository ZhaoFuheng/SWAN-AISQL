-- File: synth_googlelocal_q8.sql
-- Label: Q14
-- Question: Which Southern California campgrounds had 2021 reviews mentioning off-road trails or dirt biking?

-- Which Southern California campgrounds had 2021 reviews mentioning off-road trails or dirt biking?
DROP TABLE IF EXISTS business_description;
DROP TABLE IF EXISTS reviews;

CREATE TABLE business_description AS
SELECT *
FROM read_csv_auto('./dataset/googlelocal/business_description.csv');

CREATE TABLE reviews AS
SELECT *
FROM read_csv_auto('./dataset/googlelocal/review.csv');

WITH
campgrounds AS (
	SELECT
		bd.gmap_id,
		bd.name,
		bd.description,
		CAST(bd.num_of_reviews AS INTEGER) AS num_reviews,
		bd.state
	FROM business_description bd
	WHERE bd.num_of_reviews IS NOT NULL
	  AND CAST(bd.num_of_reviews AS INTEGER) >= 5
	  AND ai_filter('Return true only if this place is a campground, RV park, off-road recreation area, or outdoor trailhead. Otherwise return false.
Name: ' || bd.name || '
Description: ' || bd.description || '
Misc: ' || bd.misc || '
Respond with true or false only.' || e'\n Return a single yes or no, do not contain any other words.')
),
trail_reviews AS (
	SELECT
		c.gmap_id,
		c.name,
		c.num_reviews,
		c.state,
		r.time AS review_time,
		CAST(r.rating AS DOUBLE) AS rating,
		r.text
	FROM campgrounds c
	JOIN reviews r ON r.gmap_id = c.gmap_id
	WHERE r.text IS NOT NULL
	  AND LENGTH(r.text) >= 80
	  AND CAST(r.rating AS DOUBLE) >= 4
	  AND r.time ILIKE '%2021%'
	  AND (
			LOWER(r.text) LIKE '%dirt bike%'
		 OR LOWER(r.text) LIKE '%trail%'
		 OR LOWER(r.text) LIKE '%off-road%'
		 OR LOWER(r.text) LIKE '%camp%'
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
FROM trail_reviews
ORDER BY rating DESC, review_time DESC;

