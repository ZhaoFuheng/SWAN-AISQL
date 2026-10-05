-- Q3: our ThalamusDB formulation of swan_queries/Q3.sql (see make_thalamusdb_queries.py for the rule).
-- Statements end with ';' at a line end; thalamusdb_exec.py runs the plain ones as setup, a CREATE TABLE ... AS <select with
-- NLfilter> through ThalamusDB as a stage, and the last statement as the query.
CREATE TABLE loc_src AS SELECT name, state, description, coalesce(CAST(name AS VARCHAR), '') || ' in ' || coalesce(CAST(state AS VARCHAR), '') || '. Description: ' || coalesce(CAST(description AS VARCHAR), '') AS item_loc_src FROM read_csv_auto('./dataset/googlelocal/business_description.csv');
CREATE TABLE exp AS SELECT * FROM generate_series(1, 20) AS t(e);
SELECT loc_src.name, exp.e, CASE WHEN NLfilter(loc_src.item_loc_src, 'Classify the business location as Urban, Suburban or Rural: it is best described as Urban') THEN 'Urban' WHEN NLfilter(loc_src.item_loc_src, 'Classify the business location as Urban, Suburban or Rural: it is best described as Rural') THEN 'Rural' ELSE 'Suburban' END AS location_type FROM loc_src CROSS JOIN exp LIMIT 4;
