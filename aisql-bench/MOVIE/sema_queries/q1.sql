-- SemBench MOVIE q1 in Sema's dialect (s'...' semantic expressions; {col} is a column reference)
SELECT reviewId FROM reviews
WHERE s'Determine if the following movie review is clearly positive. Review: "{reviewText}".'
LIMIT 5;
