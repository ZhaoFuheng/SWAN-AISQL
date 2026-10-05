-- File: synth_googlelocal_q6.sql
-- Label: Q12
-- Question: Which open California auto repair shops earned long 2021 reviews praising honest or fast service?

-- Which open California auto repair shops earned long 2021 reviews praising honest or fast service?
DROP TABLE IF EXISTS business_description;
DROP TABLE IF EXISTS reviews;

CREATE TABLE business_description AS
SELECT *
FROM read_csv_auto('./dataset/googlelocal/business_description.csv');

CREATE TABLE reviews AS
SELECT *
FROM read_csv_auto('./dataset/googlelocal/review.csv');

WITH
auto_shops AS (
	SELECT
		bd.gmap_id,
		bd.name,
		bd.description,
		CAST(bd.num_of_reviews AS INTEGER) AS num_reviews,
		bd.state
	FROM business_description bd
	WHERE bd.num_of_reviews IS NOT NULL
	  AND CAST(bd.num_of_reviews AS INTEGER) >= 10
	  AND bd.state ILIKE 'Open%'
	  AND ai_filter('Return true only if this business is primarily an auto repair, mechanic, brake, tire, or vehicle maintenance shop. Otherwise return false.
Name: ' || bd.name || '
Description: ' || bd.description || '
Misc: ' || bd.misc || '
Respond with true or false only.' || e'\n Return a single yes or no, do not contain any other words.')
),
honest_reviews AS (
	SELECT
		s.gmap_id,
		s.name,
		s.num_reviews,
		s.state,
		r.time AS review_time,
		CAST(r.rating AS DOUBLE) AS rating,
		r.text,
		LENGTH(r.text) AS text_length
	FROM auto_shops s
	JOIN reviews r ON r.gmap_id = s.gmap_id
	WHERE r.text IS NOT NULL
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
ORDER BY rating DESC, text_length DESC;

