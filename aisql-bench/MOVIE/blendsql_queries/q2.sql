-- SemBench MOVIE q2 in BlendSQL (LLMMap over base-table or CTE columns)
SELECT reviewId FROM reviews WHERE id = 'taken_3' AND {{LLMMap('Determine if the following movie review is clearly positive.', reviews.reviewText)}} = TRUE LIMIT 5
