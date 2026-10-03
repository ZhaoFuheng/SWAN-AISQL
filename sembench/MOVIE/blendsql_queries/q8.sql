-- SemBench MOVIE q8 in BlendSQL (LLMMap over base-table or CTE columns)
WITH labeled AS (SELECT {{LLMMap('Classify the sentiment of this review as either POSITIVE or NEGATIVE.', reviews.reviewText, options=('POSITIVE','NEGATIVE'))}} AS s
  FROM reviews WHERE id = 'taken_3')
SELECT lab AS scoreSentiment, count(labeled.s) AS count
FROM (SELECT 'NEGATIVE' AS lab UNION ALL SELECT 'POSITIVE') AS labs LEFT JOIN labeled ON labeled.s = labs.lab
GROUP BY lab ORDER BY lab
