-- SemBench MOVIE q9 in BlendSQL (LLMMap over base-table or CTE columns)
SELECT reviewId, {{LLMMap('Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review', reviews.reviewText, return_type='int')}} AS reviewScore FROM reviews WHERE id = 'ant_man_and_the_wasp_quantumania'
