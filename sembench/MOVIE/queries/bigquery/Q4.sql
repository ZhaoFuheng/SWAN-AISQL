SELECT CAST(SUM(CASE WHEN AI.IF(('Determine if the following movie review is clearly positive, review: ', r.reviewText), connection_id => '<<connection>>', model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>) THEN 1 ELSE 0 END) AS FLOAT64) / COUNT(*) AS positivity_ratio
FROM movie.reviews AS r
WHERE r.id = 'taken_3';
