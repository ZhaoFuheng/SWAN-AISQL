-- SemBench MOVIE q6 in Sema's dialect (s'...' semantic expressions; {col} is a column reference).
-- Sema names each referenced column in the prompt, so the two review texts get distinct names.
SELECT id, reviewId1, reviewId2 FROM (
    SELECT r1.id, r1.reviewId AS reviewId1, r2.reviewId AS reviewId2, r1.reviewText AS review1, r2.reviewText AS review2
    FROM reviews r1 JOIN reviews r2 ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
    WHERE r1.id = 'ant_man_and_the_wasp_quantumania')
WHERE s'These two movie reviews express opposite sentiments - one is positive and the other is negative. Review 1: "{review1}" Review 2: "{review2}"'
LIMIT 10;
