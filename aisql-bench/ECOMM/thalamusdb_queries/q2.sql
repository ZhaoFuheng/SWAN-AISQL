SELECT
  image_mapping.id as id
FROM styles_details
JOIN image_mapping on styles_details.id = image_mapping.id
WHERE true
  AND NLfilter(
    local_image_path,
    'The image shows a (pair of) sports shoe(s) that feature the colors yellow and silver.'
  );
