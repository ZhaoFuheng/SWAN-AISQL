-- Q6 (table): "Which airlines have destinations in <Frankfurt | Germany | Europe>?"
--   over tampa_international_airport (row_id, Airlines, Destinations, Airport).
-- BigQuery q6a-c: AI.IF("Given destinations '" || Destinations || "' of " || Airlines ||
--   ", the airline has flights to <place>.").  SWAN: AI.IF(...) -> ai_filter(...).
-- ground_truth: q6a Frankfurt ['Discover Airlines']; q6b Germany ['Discover Airlines'];
--               q6c Europe ['British Airways','Delta Air Lines','Discover Airlines','Edelweiss Air','Virgin Atlantic'].

-- Q6a: Frankfurt
SELECT Airlines FROM tampa_international_airport
WHERE ai_filter('Given destinations ''' || Destinations || ''' of ' || Airlines
                || ', the airline has flights to Frankfurt. Base your answer ONLY on the listed destinations, not on outside knowledge about the airline.');

-- Q6b: Germany
SELECT Airlines FROM tampa_international_airport
WHERE ai_filter('Given destinations ''' || Destinations || ''' of ' || Airlines
                || ', the airline has flights to Germany. Base your answer ONLY on the listed destinations, not on outside knowledge about the airline.');

-- Q6c: Europe
SELECT Airlines FROM tampa_international_airport
WHERE ai_filter('Given destinations ''' || Destinations || ''' of ' || Airlines
                || ', the airline has flights to Europe. Base your answer ONLY on the listed destinations, not on outside knowledge about the airline.');
