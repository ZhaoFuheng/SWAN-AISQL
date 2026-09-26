SELECT 
  sentiment_label AS scoreSentiment,
  COUNT(*) AS count
FROM (
  SELECT 
    CASE WHEN AI.IF(('Determine if the following movie review is clearly positive, review: ', r.reviewText), connection_id => '<<connection>>', model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>) 
      THEN 'POSITIVE' 
      ELSE 'NEGATIVE' 
    END AS sentiment_label
  FROM movie.reviews AS r
  WHERE r.id = 'taken_3'
) AS sentiment_results
GROUP BY sentiment_label;