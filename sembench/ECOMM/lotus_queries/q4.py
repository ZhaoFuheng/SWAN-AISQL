import os
import pandas as pd
from lotus.dtype_extensions import ImageArray


def run(data_dir: str):
    # Load data
    image_mapping = pd.read_parquet(
        os.path.join(data_dir, "image_mapping.parquet")
    )
    image_mapping["images"] = ImageArray(
        image_mapping.filename.apply(
            lambda s: os.path.join(data_dir, "images", s)
        )
    )
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    )

    # Pre-filter for simple colors
    styles_details = styles_details[
        styles_details["baseColour"].isin(
            ["Black", "Blue", "Red", "White", "Orange", "Green"]
        )
    ]
    image_mapping = image_mapping[
        image_mapping["id"].astype("int").isin(styles_details["id"])
    ]

    # Process data
    processed = image_mapping.sem_extract(
        input_cols=["images"],
        output_cols={
            "category": "Extract the primary color of the product in the image. Only return the base color, nothing else."
        },
    )

    return processed[["id", "category"]]
