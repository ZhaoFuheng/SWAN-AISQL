-- SemBench MOVIE q4 in PLOP's dialect (semantic / semantic_int / semantic_string)
CREATE TABLE reviews AS SELECT * FROM read_csv_auto('data/sf_2000/Reviews.csv');

SELECT sum(CASE WHEN semantic('Determine if the following review is clearly positive. Review: ' || reviewText || '.') THEN 1 ELSE 0 END)::DOUBLE / count(*) AS positivity_ratio FROM reviews WHERE id = 'taken_3';
