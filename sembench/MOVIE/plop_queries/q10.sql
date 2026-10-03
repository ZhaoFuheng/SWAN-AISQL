-- SemBench MOVIE q10 in PLOP's dialect (semantic / semantic_int / semantic_string)
CREATE TABLE reviews AS SELECT * FROM read_csv_auto('data/sf_2000/Reviews.csv');

SELECT id AS movieId, avg(score) AS movieScore FROM (SELECT id, semantic_int('Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review: ' || reviewText) AS score FROM reviews) GROUP BY id;
