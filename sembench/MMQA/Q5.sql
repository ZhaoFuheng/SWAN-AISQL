-- Q5 (text): "Who has played a role in all the following movies: <16 movies>?"
-- ground_truth: ["Lizzy Caplan"]
-- No BigQuery reference is provided for Q5 (pure text reasoning over the movie descriptions). The 16 movies
-- are the ones whose descriptions we must jointly reason over to find the common actor.
-- SWAN: ai_agg(list(text), task) -- aggregate the 16 descriptions into ONE LLM call that names the actor
-- appearing in all of them (semantic set-intersection over the cast implied by each description).
SELECT ai_agg(
         list(text),
         'Each item is the description of a movie. Identify the single actor/actress who has played a role '
         || 'in ALL of these movies. Respond with ONLY that person''s name.'
       ) AS actor
FROM lizzy_caplan_text_data
WHERE title IN (
  'Love Is the Drug','Crashing','Cloverfield','My Best Friend''s Girl','Hot Tub Time Machine',
  'The Last Rites of Ransom Pride','Save the Date','Bachelorette','3, 2, 1... Frankie Go Boom',
  'Queens of Country','Item 47','The Night Before','Now You See Me 2','Allied','Extinction','Cobweb'
);
