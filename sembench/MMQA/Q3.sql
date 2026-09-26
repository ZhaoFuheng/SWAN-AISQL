-- Q3 (table): "Which movies are <genre>?"  over lizzy_caplan_text_data (200 movies).
-- BigQuery q3a-q3g: AI.IF(title || " is a <genre> movie given their description: " || text).
-- SWAN: AI.IF(...) -> ai_filter(...). One semantic-filter query per genre (a-g). Each returns the titles.
-- ground_truth per variant, e.g. q3a comedy -> ['Orange County','Mean Girls',...]; q3e heist -> ['Now You See Me 2'].

-- Q3a: comedies
SELECT title FROM lizzy_caplan_text_data
WHERE ai_filter(title || ' is a comedy movie given their description: ' || text);

-- Q3b: sci-fi
SELECT title FROM lizzy_caplan_text_data
WHERE ai_filter(title || ' is a sci-fi movie given their description: ' || text);

-- Q3c: romances
SELECT title FROM lizzy_caplan_text_data
WHERE ai_filter(title || ' is a romance movie given their description: ' || text);

-- Q3d: horror
SELECT title FROM lizzy_caplan_text_data
WHERE ai_filter(title || ' is a horror movie given their description: ' || text);

-- Q3e: heist movies
SELECT title FROM lizzy_caplan_text_data
WHERE ai_filter(title || ' is a heist movie given their description: ' || text);

-- Q3f: romantic comedies
SELECT title FROM lizzy_caplan_text_data
WHERE ai_filter(title || ' is a romantic comedy movie given their description: ' || text);

-- Q3g: biographical comedies
SELECT title FROM lizzy_caplan_text_data
WHERE ai_filter(title || ' is a biographical comedy movie given their description: ' || text);
