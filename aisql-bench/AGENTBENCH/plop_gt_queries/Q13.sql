-- File: synth_googlelocal_q7.sql
-- Label: Q13
-- Question: Which open California nail salons received detailed 2021 reviews highlighting manicure or brow results?

-- Which open California nail salons received detailed 2021 reviews highlighting manicure or brow results?
DROP TABLE IF EXISTS business_description;
DROP TABLE IF EXISTS reviews;

CREATE TABLE business_description AS
SELECT *
FROM read_csv_auto('./test/semantic/agent_bench/dataset/googlelocal/business_description.csv');

CREATE TABLE reviews AS
SELECT *
FROM read_csv_auto('./test/semantic/agent_bench/dataset/googlelocal/review.csv');

WITH
nail_salons AS (
	SELECT
		bd.gmap_id,
		bd.name,
		bd.description,
		CAST(bd.num_of_reviews AS INTEGER) AS num_reviews,
		bd.state
	FROM business_description bd
	WHERE bd.num_of_reviews IS NOT NULL
	  AND CAST(bd.num_of_reviews AS INTEGER) >= 3
	  AND SEMANTIC('Return true only if this business primarily offers nail salon, manicure, pedicure, brow, or spa services. Otherwise return false.
Name: {bd.name}
Description: {bd.description}
Misc: {bd.misc}
Respond with true or false only.')
),
beauty_feedback AS (
	SELECT
		n.gmap_id,
		n.name,
		n.num_reviews,
		n.state,
		r.time AS review_time,
		CAST(r.rating AS DOUBLE) AS rating,
		r.text
	FROM nail_salons n
	JOIN reviews r ON r.gmap_id = n.gmap_id
	WHERE r.text IS NOT NULL
	  AND LENGTH(r.text) >= 120
	  AND CAST(r.rating AS DOUBLE) >= 4.5
	  AND r.time ILIKE '%2021%'
	  AND (
			LOWER(r.text) LIKE '%nail%'
		 OR LOWER(r.text) LIKE '%mani%'
		 OR LOWER(r.text) LIKE '%pedi%'
		 OR LOWER(r.text) LIKE '%brow%'
		 OR LOWER(r.text) LIKE '%dip%'
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
FROM beauty_feedback
ORDER BY rating DESC, review_time DESC;

