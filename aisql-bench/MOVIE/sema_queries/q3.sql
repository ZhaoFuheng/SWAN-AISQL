-- SemBench MOVIE q3 in Sema's dialect (s'...' semantic expressions; {col} is a column reference)
SELECT count(*) AS positive_review_cnt FROM reviews
WHERE id = 'taken_3'
  AND s'Determine if the following review is clearly positive. Review: {reviewText}';
