import os
import pandas as pd
from lotus.dtype_extensions import ImageArray

def run(data_dir: str):
    # Load data
    styles_details = pd.read_parquet(os.path.join(data_dir, 'styles_details.parquet'))
    image_mapping = pd.read_parquet(os.path.join(data_dir, 'image_mapping.parquet'))
    styles_details['id'] = styles_details['id'].astype('int')
    image_mapping['image_id'] = image_mapping['id'].astype('int')
    styles_details = styles_details.merge(image_mapping, left_on='id', right_on='image_id', suffixes=('', '_img'))
    styles_details['images']  = ImageArray(styles_details.filename.apply(lambda s: os.path.join(data_dir, 'images', s)))

    # Filter data
    filtered = styles_details.sem_filter('''
        You will receive a description of what a customer is looking for together with an image and a textual description of the product.
        Determine if they both match.
    
        I am looking for a running shirt for men with a round neck and short sleeves,
        preferably in blue or black, but not bright colors like white.
        Also definitely not green.
        It should be suitable for outdoor running in warm weather.
        If the t-shirt is not green, it should at least feature a striped design.

        The product has the following image {images} and textual description {productDisplayName} {productDescriptors}
    ''')

    return filtered['id']
