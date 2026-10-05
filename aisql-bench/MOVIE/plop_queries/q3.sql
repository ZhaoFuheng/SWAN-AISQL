-- SemBench MOVIE q3 in PLOP's dialect (semantic / semantic_int / semantic_string)
CREATE TABLE reviews AS SELECT * FROM read_csv_auto('data/sf_2000/Reviews.csv');

SELECT count(*) AS positive_review_cnt FROM reviews WHERE id = 'taken_3' AND semantic('Determine if the following review is clearly positive. Review: ' || reviewText);
