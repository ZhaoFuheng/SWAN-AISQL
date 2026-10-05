import os
import pandas as pd
from lotus.dtype_extensions import ImageArray


def run(data_dir: str):
    # Load data
    styles_details = pd.read_parquet(os.path.join(data_dir, 'styles_details.parquet'))
    image_mapping = pd.read_parquet(os.path.join(data_dir, 'image_mapping.parquet'))

    styles_details['id'] = styles_details['id'].astype('int')
    image_mapping['image_id'] = image_mapping['id'].astype('int')
    
    # Pre-filter data
    styles_details = styles_details[styles_details.apply(lambda row: row['masterCategory']['typeName'] in ('Accessories', 'Apparel', 'Footwear'), axis=1)]
    image_mapping = image_mapping[image_mapping['image_id'].isin(styles_details['id'])]
    styles_details = styles_details.merge(image_mapping, left_on='id', right_on='image_id', suffixes=('', '_img'))
    styles_details['images']  = ImageArray(styles_details.filename.apply(lambda s: os.path.join(data_dir, 'images', s)))

    # Semantic filter
    styles_details = styles_details.sem_filter('Does the following description describe a product from either Adidas or Puma? {productDisplayName} {productDescriptors}')

    # Extract JSON data
    processed = styles_details.sem_extract(
        input_cols=["id", "productDisplayName", "productDescriptors", "images"],
        output_cols={
            "generated_json": """
            You are given a product description and an image of the product as well as the product id.
            The product contains a fashion item (clothing, shoes, accessories, etc).
            There might be multiple fashion items in the image, especially when a model is presenting them.
            If this is the case, focus only on the primary fashion item and use the description to determine which item in the image is of interest.

            For each product, generate the following JSON:
            ```
            {
              "id": <product id> (integer),
              "brand": <extract the brand name from the description and/or image. use lower-case letters for the brand name>",
              "category": <classify the images into 'accessories', 'apparel', 'footwear'>
            }
            ```

            Output the json in a single line.
            Keep the order of the keys in the JSON as given in the description.
            Do not use spaces between { or keys and values in the JSON, i.e., do no use spaces anywhere in the JSON structure.
            Use normal quotes in the JSON; do not use single quotes.
            """
        },
    )

    return processed["generated_json"].rename("id")
