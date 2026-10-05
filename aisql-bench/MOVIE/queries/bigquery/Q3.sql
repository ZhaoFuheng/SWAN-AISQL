SELECT COUNT(*) AS positive_review_cnt
FROM movie.reviews AS r
WHERE r.id = 'taken_3' AND AI.IF(('Determine if the following movie review is clearly positive, review: ', r.reviewText), connection_id => '<<connection>>', model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>);
