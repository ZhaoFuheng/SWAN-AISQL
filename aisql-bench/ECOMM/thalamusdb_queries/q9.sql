SELECT
  img1.id || '-' || img2.id AS id
FROM styles_details s1, styles_details s2, image_mapping img1, image_mapping img2
WHERE true
  -- Pre-filter both image tables with conditions on styles_details
  AND s1.baseColour IN ('Black', 'Blue', 'Red', 'White', 'Orange', 'Green')
  AND s1.colour1 = ''
  AND s1.colour2 = ''
  AND s1.price < 800
  AND s1.id = img1.id
  AND s2.baseColour IN ('Black', 'Blue', 'Red', 'White', 'Orange', 'Green')
  AND s2.colour1 = ''
  AND s2.colour2 = ''
  AND s2.price < 800
  AND s2.id = img2.id
  -- Semantic join
  AND img1.id <> img2.id
  AND NLjoin(img1.local_image_path, img2.local_image_path,
     '''
     Determine whether both images display objects of the same category
     (e.g., both are shoes, both are bags, etc.) and whether these objects
     share the same dominant surface color. Disregard any logos, text, or
     printed graphics on the objects. There might be other objects in the
     images. Only focus on the main object. Base your comparison solely on
     object type and overall surface color.
     '''
)
;