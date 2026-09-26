-- Q1 (table + text): "Who is the director of the movie that has Ben Piazza in the role of Bob Whitewood?"
-- ground_truth: ["Michael Ritchie"]
-- BigQuery: AI.GENERATE(extract director from text) over ben_piazza_text_data, JOIN ben_piazza on Title,
--           WHERE Role = "Bob Whitewood".  SWAN: AI.GENERATE(...).result -> ai_complete(...).
-- We filter to the single Bob-Whitewood movie FIRST (via the join), then extract the director from its
-- description -- 1 LLM call instead of one per row (result-identical; the AI runs only on qualifying rows).
SELECT ai_complete(
         'Extract the director name from the following movie description. '
         || 'Respond with ONLY the director''s full name, nothing else.' || chr(10) || chr(10) || td.text
       ) AS director
FROM ben_piazza bp
JOIN ben_piazza_text_data td ON bp.Title = td.title
WHERE bp.Role = 'Bob Whitewood';
