import os
import pandas as pd

def run(data_dir: str):
    # Load data
    styles_details = pd.read_parquet(os.path.join(data_dir, 'styles_details.parquet'))

    # Filter data
    # TODO: productDescriptors contains sub-columns and should only access 'productDescriptors.description.value'
    filtered = styles_details.sem_filter('The product is a backpack from Reebok: {productDisplayName} {productDescriptors}')

    return filtered['id']
