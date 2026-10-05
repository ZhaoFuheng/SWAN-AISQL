SELECT
  styles_details.id || '-' || image_mapping.id as id
FROM styles_details, image_mapping
WHERE true
  AND NLjoin(styles_details.full_product_description, image_mapping.local_image_path, 'The image fits the description')
  AND character_length(styles_details.description) >= 3000
;
