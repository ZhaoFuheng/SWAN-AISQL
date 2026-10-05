SELECT t.Airlines AS 'Airlines', i.image_filename AS image_id
FROM tampa_airport t, images i
WHERE NLjoin(t.Airlines, i.image_filepath, 'the image shows the airline logo');
