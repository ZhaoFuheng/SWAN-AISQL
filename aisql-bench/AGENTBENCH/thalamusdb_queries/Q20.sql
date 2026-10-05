-- Q20: our ThalamusDB formulation of swan_queries/Q20.sql (see make_thalamusdb_queries.py for the rule).
-- Statements end with ';' at a line end; thalamusdb_exec.py runs the plain ones as setup, a CREATE TABLE ... AS <select with
-- NLfilter> through ThalamusDB as a stage, and the last statement as the query.
CREATE TABLE yelp_business AS SELECT * FROM read_csv_auto('./dataset/yelp/yelp_business_csv/business.csv', maximum_line_size=1048576);
CREATE TABLE yelp_review AS SELECT * FROM read_csv_auto('./dataset/yelp/review.csv', maximum_line_size=1048576);
CREATE TABLE yelp_tip AS SELECT * FROM read_csv_auto('./dataset/yelp/tip.csv', maximum_line_size=1048576);
CREATE TABLE reviews_link AS SELECT 'businessid_' || regexp_extract(r.business_ref, '([0-9]+)$') AS business_id, r.rating, r.date AS review_date FROM yelp_review r WHERE r.business_ref LIKE 'businessref_%';
CREATE TABLE tips_link AS SELECT 'businessid_' || regexp_extract(t.business_ref, '([0-9]+)$') AS business_id, t.date AS tip_date FROM yelp_tip t WHERE t.business_ref LIKE 'businessref_%';
CREATE TABLE yelp_src AS SELECT yb.*, 'Business ID: ' || coalesce(CAST(yb.business_id AS VARCHAR), '') || E'\n' || 'Attributes: ' || coalesce(CAST(yb.attributes AS VARCHAR), '') || E'\n' || 'Description: ' || coalesce(CAST(yb.description AS VARCHAR), '') AS item_yelp_src FROM yelp_business yb;
SELECT yelp_src.business_id, rl.review_date, rl.rating FROM yelp_src JOIN reviews_link rl ON rl.business_id = yelp_src.business_id LEFT JOIN tips_link tl ON tl.business_id = yelp_src.business_id WHERE rl.review_date ILIKE '%2018%' AND rl.rating >= 4 AND yelp_src.attributes ILIKE '%"BikeParking": "True"%' AND tl.tip_date ILIKE '%2018%' AND yelp_src.description ILIKE '%Philadelphia%' AND NLfilter(yelp_src.item_yelp_src, 'Return true if the business offers customer parking (lot, garage, street, or valet) or any form of bike parking. Use the provided attributes and description to decide.') ORDER BY rl.rating DESC, rl.review_date DESC;
