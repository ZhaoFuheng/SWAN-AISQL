SELECT
  p1.id || '-' || p2.id AS "id"
FROM styles_details p1, styles_details p2
WHERE true
  AND NLjoin(p1.full_product_description, p2.full_product_description, 'You will be given two product descriptions. Do both product descriptions describe products of the same category from the same brand, e.g., both are t-shirts from Adidas?')
  AND p1.price <= 500
  AND p2.price <= 500
;
