-- SemBench MOVIE q3 in BlendSQL (LLMMap over base-table or CTE columns)
SELECT count(*) AS positive_review_cnt FROM reviews WHERE id = 'taken_3' AND {{LLMMap('Determine if the following review is clearly positive.', reviews.reviewText)}} = TRUE
