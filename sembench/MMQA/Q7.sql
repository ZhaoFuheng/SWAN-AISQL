-- Q7 (table + IMAGE): "For each airline with destinations in Europe, find its logo if one exists."
--   tampa_international_airport(Airlines,Destinations) x images(image_filepath).
--   ground_truth: [['British Airways','cwnc...png'], ['Delta Air Lines','1sz0...png'], ...].
-- BigQuery q7: AI.IF("...Airline: "||t.Airlines||", Image: "||i.uri).  SWAN: composes Q6c's TEXT filter
--   (airlines in Europe) with an IMAGE ai_filter (logo match) via ai_image(path).
WITH europe_airlines AS (
  SELECT Airlines FROM tampa_international_airport
  WHERE ai_filter('Given destinations ''' || Destinations || ''' of ' || Airlines
                  || ', the airline has flights to Europe.')
)
SELECT e.Airlines, i.image_filename
FROM europe_airlines e, images i
WHERE ai_filter('Does this image show the logo of the airline "' || e.Airlines || '"? '
                || ai_image(i.image_filepath))
ORDER BY e.Airlines;
