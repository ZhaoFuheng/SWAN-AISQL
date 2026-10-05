-- SemBench MOVIE q4 in BlendSQL (LLMMap over base-table or CTE columns)
WITH flagged AS (SELECT {{LLMMap('Determine if the following review is clearly positive.', reviews.reviewText)}} AS pos
  FROM reviews WHERE id = 'taken_3')
SELECT CAST(sum(CASE WHEN pos = TRUE THEN 1 ELSE 0 END) AS DOUBLE) / count(*) AS positivity_ratio FROM flagged
