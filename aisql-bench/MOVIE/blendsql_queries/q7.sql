-- SemBench MOVIE q7 in BlendSQL (LLMMap over base-table or CTE columns)
WITH pairs AS (SELECT r1.id AS id, r1.reviewId AS reviewId1, r2.reviewId AS reviewId2,
  'Review 1: "' || r1.reviewText || '" Review 2: "' || r2.reviewText || '"' AS pair_text
  FROM reviews r1 JOIN reviews r2 ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
  WHERE r1.id = 'ant_man_and_the_wasp_quantumania')
SELECT id, reviewId1, reviewId2 FROM pairs WHERE {{LLMMap('These two movie reviews express opposite sentiments - one is positive and the other is negative.', pairs.pair_text)}} = TRUE
