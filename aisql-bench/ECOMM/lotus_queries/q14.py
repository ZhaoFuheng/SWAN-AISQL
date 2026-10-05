from operator import itemgetter
import os
import pandas as pd
from lotus.dtype_extensions import ImageArray

def run(data_dir: str):
    # NOTE: Approximate policy with cascade_args is NOT supported for this query.
    # LOTUS does not support embedding-based optimization for multi-column semantic joins
    # (joins that reference multiple columns like {images} AND {productDescriptors} simultaneously).
    # Cascade optimization only works for single-column joins.

    # Load data
    styles_details = pd.read_parquet(os.path.join(data_dir, 'styles_details.parquet'))
    image_mapping = pd.read_parquet(os.path.join(data_dir, 'image_mapping.parquet'))
    image_mapping['images']  = ImageArray(image_mapping.filename.apply(lambda s: os.path.join(data_dir, 'images', s)))

    # Pre-filter
    styles_details = styles_details[styles_details.apply(lambda row: row['price'] < 130, axis=1)]

    # Semantic filter
    image_mapping = image_mapping.sem_filter('The image {images} depicts white socks')

    # Semantic join
    processed = styles_details.sem_join(image_mapping, '''
     The image {images} fits the description: {productDisplayName} {productDescriptors}
    ''')

    # Semantic "best-match"
    processed = processed.sem_topk('''
     The image {images} fits the description: {productDisplayName} {productDescriptors}
    ''', K=1, group_by='id:left')

    processed['id'] = processed['id:right'].astype('str')
    return processed['id']
