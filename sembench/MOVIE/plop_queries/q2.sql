-- SemBench MOVIE q2 in PLOP's dialect (semantic / semantic_int / semantic_string)
CREATE TABLE reviews AS SELECT * FROM read_csv_auto('data/sf_2000/Reviews.csv');

SELECT reviewId FROM reviews WHERE id = 'taken_3' AND semantic('Determine if the following movie review is clearly positive. Review: "' || reviewText || '".') LIMIT 5;
