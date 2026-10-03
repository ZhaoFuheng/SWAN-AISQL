-- SemBench MOVIE q8 in PLOP's dialect (semantic / semantic_int / semantic_string)
CREATE TABLE reviews AS SELECT * FROM read_csv_auto('data/sf_2000/Reviews.csv');

WITH labeled AS (SELECT upper(trim(semantic_string('Classify the sentiment of this review as either POSITIVE or NEGATIVE. Answer with exactly one of the two words. Review: ' || reviewText))) AS s FROM reviews WHERE id = 'taken_3')
SELECT lab AS scoreSentiment, count(labeled.s) AS count
FROM (VALUES ('NEGATIVE'), ('POSITIVE')) AS labs(lab) LEFT JOIN labeled ON labeled.s = labs.lab
GROUP BY lab ORDER BY lab;
