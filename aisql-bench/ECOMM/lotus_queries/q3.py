import os
import pandas as pd


def run(data_dir: str):
    # Load data
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    )

    # Apply map
    # TODO: productDescriptors contains sub-columns and should only access 'productDescriptors.description.value'
    processed = styles_details.sem_extract(
        input_cols=["productDisplayName", "productDescriptors"],
        output_cols={
            "category": "Extract the brand name from the following product description."
        },
    )

    return processed[["id", "category"]]
