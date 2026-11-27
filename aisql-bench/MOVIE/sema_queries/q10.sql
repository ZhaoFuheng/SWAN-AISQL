-- SemBench MOVIE q10 in Sema's dialect (s'...' semantic expressions; {col} is a column reference).
-- Sema's map returns text: the score is its first digit 1-5.
SELECT id AS movieId, avg(score) AS movieScore
FROM (SELECT id,
             TRY_CAST(regexp_extract(s'Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review: {reviewText}', '[1-5]') AS INTEGER) AS score
      FROM reviews)
GROUP BY id;
