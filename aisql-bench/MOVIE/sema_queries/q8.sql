-- SemBench MOVIE q8 in Sema's dialect (s'...' semantic expressions; {col} is a column reference).
-- Sema has no classify operator: the label comes from a semantic map, read with the label match below.
-- Sema's s'...' keeps a doubled '' as two quote characters, so the labels are quoted with " instead.
WITH labeled AS (
    SELECT regexp_extract(upper(s'Classify the sentiment of this review as either "POSITIVE" or "NEGATIVE". Review: {reviewText}'),
                          'POSITIVE|NEGATIVE') AS s
    FROM reviews WHERE id = 'taken_3')
SELECT lab AS scoreSentiment, count(labeled.s) AS count
FROM (VALUES ('NEGATIVE'), ('POSITIVE')) AS labs(lab)
LEFT JOIN labeled ON labeled.s = labs.lab
GROUP BY lab ORDER BY lab;
