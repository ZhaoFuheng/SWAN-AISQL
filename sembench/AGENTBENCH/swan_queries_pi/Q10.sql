-- File: synth_googlelocal_q4.sql
-- Label: Q10
-- Question: Which GoogleLocal massage therapy businesses receive detailed high-rated reviews?

-- Which GoogleLocal massage therapy businesses receive detailed high-rated reviews?
DROP TABLE IF EXISTS business_description;
DROP TABLE IF EXISTS reviews;

CREATE TABLE business_description AS
SELECT *
FROM read_csv_auto('./dataset/googlelocal/business_description.csv');

CREATE TABLE reviews AS
SELECT *
FROM read_csv_auto('./dataset/googlelocal/review.csv');

WITH
massage_businesses AS (
	SELECT
		bd.gmap_id,
		bd.name,
		bd.description,
		bd.num_of_reviews,
		bd.state
	FROM business_description bd
	WHERE bd.num_of_reviews IS NOT NULL
	  AND CAST(bd.num_of_reviews AS INTEGER) >= 15
	  AND bd.state LIKE 'Open%'
	  AND (lower(ai_complete('Return true if this business primarily offers massage therapy or spa treatments. Otherwise return false.
Name: ' || bd.name || '
Description: ' || bd.description || '
Misc: ' || bd.misc || '
Number of Reviews: ' || bd.num_of_reviews || '
Respond with true or false only.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
),
positive_reviews AS (
	SELECT
		mb.gmap_id,
		mb.name,
		CAST(mb.num_of_reviews AS INTEGER) AS business_review_count,
		mb.state,
		r.time AS review_time,
		CAST(r.rating AS DOUBLE) AS rating,
		r.text
	FROM massage_businesses mb
	JOIN reviews r ON r.gmap_id = mb.gmap_id
	WHERE CAST(r.rating AS DOUBLE) >= 4.8
	  AND r.text IS NOT NULL
	  AND LENGTH(r.text) >= 150
	  AND (
			POSITION('2021' IN r.time) > 0
			OR POSITION('2022' IN r.time) > 0
			OR POSITION('2023' IN r.time) > 0
			OR POSITION('2024' IN r.time) > 0
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
ORDER BY rating DESC, review_time DESC;
