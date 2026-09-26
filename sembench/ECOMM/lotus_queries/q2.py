import os
import pandas as pd
from lotus.dtype_extensions import ImageArray

def run(data_dir: str):
    # Load data
    image_mapping = pd.read_parquet(os.path.join(data_dir, 'image_mapping.parquet'))
    image_mapping['images']  = ImageArray(image_mapping.filename.apply(lambda s: os.path.join(data_dir, 'images', s)))

    # Filter data
    filtered = image_mapping.sem_filter('The image shows a (pair of) sports shoe(s) that feature the colors yellow and silver. {images}')

    return filtered['id']
