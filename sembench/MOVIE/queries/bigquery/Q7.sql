SELECT r1.id, r1.reviewId, r2.reviewId
FROM movie.reviews AS r1
JOIN movie.reviews AS r2
ON r1.id = r2.id AND r1.reviewId <> r2.reviewId
WHERE r1.id = 'ant_man_and_the_wasp_quantumania' AND AI.IF(
    ('These two movie reviews express opposite sentiments - one is positive and the other is negative. Review 1: ', r1.reviewText, ', Review 2: ', r2.reviewText), 
    connection_id => '<<connection>>', model_params => JSON '{"labels":{"query_uuid": "<<query_id>>"}, "generation_config":{"thinking_config": {"thinking_budget": <<thinking_budget>>}}}' <<other_params>>
);