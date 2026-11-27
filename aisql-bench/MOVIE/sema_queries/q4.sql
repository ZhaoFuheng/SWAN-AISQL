-- SemBench MOVIE q4 in Sema's dialect (s'...' semantic expressions; {col} is a column reference).
-- A semantic string inside CASE WHEN is a map in Sema (free text cast to BOOLEAN), so the ratio is two counts
-- with the predicate as a filter, which is the same quantity.
SELECT (SELECT count(*) FROM reviews
        WHERE id = 'taken_3'
          AND s'Determine if the following review is clearly positive. Review: {reviewText}.')::DOUBLE
       / (SELECT count(*) FROM reviews WHERE id = 'taken_3') AS positivity_ratio;
