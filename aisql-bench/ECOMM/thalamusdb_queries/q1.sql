SELECT
  id
FROM styles_details
WHERE true
  AND NLfilter(
    full_product_description,
    'The product is a backpack from Reebok'
  );
