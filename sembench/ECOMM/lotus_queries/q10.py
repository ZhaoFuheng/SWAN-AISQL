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
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    )
    image_mapping = pd.read_parquet(
        os.path.join(data_dir, "image_mapping.parquet")
    )

    # Pre-filter data
    styles_details = styles_details[
        (styles_details["baseColour"].isin(["Black", "Blue", "Red", "White"]))
        & (styles_details["price"] <= 1000)
    ]
    styles_details["id"] = styles_details["id"].astype("int")
    image_mapping["image_id"] = image_mapping["id"].astype("int")
    styles_details = styles_details.merge(
        image_mapping, left_on="id", right_on="image_id", suffixes=("", "_img")
    )
    styles_details["images"] = ImageArray(
        styles_details.filename.apply(
            lambda s: os.path.join(data_dir, "images", s)
        )
    )

    # Filters
    footwear = styles_details.sem_filter(
        "The image depicts a (pair of) shoe(s), sandal(s), flip-flop(s). If there are multiple products in the picture, always refer to the most promiment one. {images}"
    )
    bottomwear = styles_details.sem_filter(
        "The image depicts a piece of apparel that can be worn on the lower part of the body, like pants, shorts, skirts, ... If there are multiple products in the picture, always refer to the most promiment one. {images}"
    )
    topwear = styles_details.sem_filter(
        "The image depicts a piece of apparel that can be worn on the upper part of the body, like t-shirts, shirts, pullovers, hoodies, but still require some sort of clothing on the lower body, which means, e.g., not a dress. If there are multiple products in the picture, always refer to the most promiment one. {images}"
    )

    # Perform joins
    processed = footwear.sem_join(
        bottomwear,
        """
        The images depict products with the same primary base color, e.g., both are black, both are white, and both products are from the same brand.
        The description of the first product is {productDisplayName:left} {productDescriptors:left} and the image of the first product is {images:left}.
        The description of the second product is {productDisplayName:right} {productDescriptors:right} and the image of the second product is {images:right}
        """
    )
    processed.rename(
        mapper=lambda x: x.replace(":", "_"), axis="columns", inplace=True
    )
    processed = processed.sem_join(
        topwear,
        """
        The images depict products with the same primary base color, e.g., both are black, both are white, and both products are from the same brand.
        The description of the first product is {productDisplayName_right} {productDescriptors_right} and the image of the first product is {images_right}.
        The description of the second product is {productDisplayName} {productDescriptors} and the image of the second product is {images}
        """
    )

    processed["id"] = (
        processed["id_left"].astype("str")
        + "-"
        + processed["id_right"].astype("str")
        + "-"
        + processed["id"].astype("str")
    )
    return processed["id"]
