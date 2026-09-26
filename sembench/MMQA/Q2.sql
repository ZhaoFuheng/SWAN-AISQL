-- Q2 (table + IMAGE): "Identify the images containing logos, if available, for each racetrack in which
--   A.P. Warrior was a contender. [q2b: What is the color of each logo?]"
--   ap_warrior(...,Track) CROSS JOIN images(image_filepath).  ground_truth: [[ID, image.png], ...] (+ color).
-- BigQuery q2a: AI.IF("...Racetrack: "||t.Track||", Image: "||i.uri).  SWAN: ai_filter(text || ai_image(path)).
-- SWAN now has vision: ai_image(path) sends the image to the multimodal model. Distinct racetracks only.

-- Q2a: for each racetrack, which images are its logo?
SELECT DISTINCT w.ID AS id, i.image_filename
FROM ap_warrior w, images i
WHERE ai_filter('Does this image show the logo of the racetrack "' || w.Track || '"? ' || ai_image(i.image_filepath))
ORDER BY id;

-- Q2b: + the logo's dominant color.
SELECT DISTINCT w.ID AS id, i.image_filename,
       ai_complete('What is the single dominant color of the logo in this image? Answer with one word: '
                   || ai_image(i.image_filepath)) AS color
FROM ap_warrior w, images i
WHERE ai_filter('Does this image show the logo of the racetrack "' || w.Track || '"? ' || ai_image(i.image_filepath))
ORDER BY id;
