SELECT t.ID AS 'ID', image_filename AS image_id
FROM ap_warrior t, images i
WHERE NLjoin(t.Track, i.image_filepath, 'the image shows the logo of the horse racetrack');
