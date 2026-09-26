-- Q4 (table): "Categorize the movies in the table by their genre. If a movie belongs to multiple genres,
--   list it under each applicable genre."  over lizzy_caplan_text_data.
-- ground_truth: {'comedy': [...], 'sci-fi': [...], ...}
-- BigQuery: AI.GENERATE(prompt, output_schema="genres ARRAY<STRING>").genres, then UNNEST + GROUP BY genre.
-- SWAN: ai_complete returns a COMMA-SEPARATED genre list (this DuckDB fork has no json extension, so we
--   avoid JSON arrays); DuckDB string_split + unnest fans it out to one (movie, genre) row per genre.
-- (Restricted to the 24 base movies, matching the BigQuery reference / the ground-truth universe.)
WITH movie_genres AS (
  SELECT title,
         ai_complete(
           'List every applicable film genre for this movie based on its description. '
           || 'Answer as a COMMA-SEPARATED list of short lowercase genre labels only '
           || '(e.g. "comedy, drama, sci-fi"), no other text. Description: ' || text
         ) AS genres_csv
  FROM lizzy_caplan_text_data
  WHERE title IN (
    'Orange County','Mean Girls','Love Is the Drug','Crashing','Cloverfield','My Best Friend''s Girl',
    'Crossing Over','Hot Tub Time Machine','The Last Rites of Ransom Pride','127 Hours','High Road',
    'Save the Date','Bachelorette','3, 2, 1... Frankie Go Boom','Queens of Country','Item 47',
    'The Interview','The Night Before','Now You See Me 2','Allied','The Disaster Artist','Extinction',
    'The People We Hate at the Wedding','Cobweb'
  )
)
SELECT lower(trim(genre)) AS genre,
       string_agg(title, ', ' ORDER BY title) AS movies_in_genre
FROM (SELECT title, unnest(string_split(genres_csv, ',')) AS genre FROM movie_genres)
WHERE trim(genre) <> ''
GROUP BY 1
ORDER BY 1;
