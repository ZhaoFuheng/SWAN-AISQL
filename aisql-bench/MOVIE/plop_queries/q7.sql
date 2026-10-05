-- SemBench MOVIE q7 in PLOP's dialect (semantic / semantic_int / semantic_string)
CREATE TABLE reviews AS SELECT * FROM read_csv_auto('data/sf_2000/Reviews.csv');

SELECT r1.id, r1.reviewId AS reviewId1, r2.reviewId AS reviewId2 FROM reviews r1 JOIN reviews r2 ON r1.id = r2.id AND r1.reviewId <> r2.reviewId WHERE r1.id = 'ant_man_and_the_wasp_quantumania' AND semantic('These two movie reviews express opposite sentiments - one is positive and the other is negative. Review 1: "' || r1.reviewText || '" Review 2: "' || r2.reviewText || '"');
