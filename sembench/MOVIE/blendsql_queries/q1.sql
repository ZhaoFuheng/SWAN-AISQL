-- SemBench MOVIE q1 in BlendSQL (LLMMap over base-table or CTE columns)
SELECT reviewId FROM reviews WHERE {{LLMMap('Determine if the following movie review is clearly positive.', reviews.reviewText)}} = TRUE LIMIT 5
