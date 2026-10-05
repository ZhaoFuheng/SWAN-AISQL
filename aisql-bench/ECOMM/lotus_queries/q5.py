from operator import itemgetter
import os
import pandas as pd

def run(data_dir: str):
    # Load data
    styles_details = pd.read_parquet(os.path.join(data_dir, 'styles_details.parquet'))

    # Preprocess data
    styles_details = styles_details[styles_details.apply(lambda row: row['masterCategory']['typeName'] == 'Apparel' and row['subCategory']['typeName'] not in ['Saree', 'Apparel Set', 'Loungewear and Nightwear'], axis=1)]
    
    # Apply classification
    # TODO: productDescriptors contains sub-columns and should only access 'productDescriptors.description.value'
    processed = styles_details.sem_map('''
        You are given a description of a product. Your task is to classify the product
        into one of the following categories: 
        (1) Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down
                 over the legs, and often consist of a bodice attached to a skirt.
        (2) Bottomwear: Bottomwear refers to clothing worn on the lower part of the body,
                 such as trousers, jeans, skirts, shorts, and leggings.
        (3) Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric,
                 designed to provide comfort and warmth.
        (4) Topwear: Topwear refers to clothing worn on the upper part of the body,
                 such as shirts, blouses, t-shirts, and jackets
        (5) Innerwear: Innerwear refers to clothing worn beneath outer garments,
                 typically close to the skin, such as underwear, bras, and undershirts.
        When classifying the product, only output the category name, nothing more.

        The product description is as follows: {productDisplayName} {productDescriptors}
        ''',
        suffix='category'
    )

    return processed[['id', 'category']]
