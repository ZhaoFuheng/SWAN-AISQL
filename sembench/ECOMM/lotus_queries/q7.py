from operator import itemgetter
import os
import pandas as pd

def run(data_dir: str):
    # Configure LOTUS for text-only join if approximate policy is enabled
    if '_configure_lotus' in globals() and '_policy' in globals() and globals()['_policy'] == 'approximate':
        globals()['_configure_lotus']('text')

    # Load data
    styles_details = pd.read_parquet(os.path.join(data_dir, 'styles_details.parquet'))

    # Pre-filter data
    styles_details = styles_details[styles_details['price'] <= 500]

    # Reset index for approximate policy
    if '_policy' in globals() and globals()['_policy'] == 'approximate':
        styles_details = styles_details.reset_index(drop=True)

    # Self-join
    join_instruction = '''
     You will be given two product descriptions. Do both product descriptions describe
     products of the same category from the same brand, e.g., both are t-shirts from Adidas?

     The first product description is:
     {productDisplayName:left} - {productDescriptors:left}

     The second product description is:
     {productDisplayName:right} - {productDescriptors:right}
    '''

    if '_cascade_args' in globals() and '_policy' in globals() and globals()['_policy'] == 'approximate':
        processed = styles_details.sem_join(
            styles_details,
            join_instruction,
            cascade_args=globals()['_cascade_args']
        )
    else:
        processed = styles_details.sem_join(styles_details, join_instruction)

    processed['id'] = processed['id:left'].astype('str') + '-' + processed['id:right'].astype('str')
    return processed['id']
