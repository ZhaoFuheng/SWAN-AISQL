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
    footwear = styles_details.sem_filter("""
        You will receive an image and a description of a product.
        Determine whether the product can be worn on the feet, like shoes, sandals, flip-flops, ...
        The predominant color of the depicted product should be black.
        If there are multiple products in the picture, always refer to the most promiment one.
        The description of the product is as follows:
        {productDisplayName} {productDescriptors} {images}""")
    bottomwear = styles_details.sem_filter("""
        You will receive an image and a description of a product.
        Determine whether the product can be worn on the lower part of the body, like pants, shorts, skirts, ...
        The predominant color of the depicted product should be black.
        Do not consider swimwear.
        If there are multiple products in the picture, always refer to the most promiment one.
        The description of the product is as follows:
        {productDisplayName} {productDescriptors} {images}""")
    topwear = styles_details.sem_filter("""
        You will receive an image and a description of a product.
        Determine whether the product can be worn on the upper part of the body, like t-shirts, shirts, pullovers, hoodies, but still require some sort of clothing on the lower body, which means, e.g., not a dress.
        The predominant color of the depicted product should be black.
        Do not consider swimwear.
        If there are multiple products in the picture, always refer to the most promiment one.
        The description of the product is as follows:
        {productDisplayName} {productDescriptors} {images}""")
    accessories = styles_details[styles_details["price"] <= 500]
    accessories = accessories.sem_filter("""
        You will receive an image and a description of a product.
        Determine whether the product a watch or some jewellery or a bag.
        A bag might be a handbag or a (gym) backpack or some other type of bag.
        If there are multiple products in the picture, always refer to the most promiment one.
        The description of the product is as follows:
        {productDisplayName} {productDescriptors} {images}""")

    # Perform joins
    processed_1 = footwear.sem_join(
        accessories,
        """
     You will receive a description and an image of two products. Determine whether they are from the same brand.
     The description of the first product is as follows: {productDisplayName:left} {productDescriptors:left}.
     And the image of the first product is {images:left}.
     The description of the second product is as follows: {productDisplayName:right} {productDescriptors:right}.
     And the image of the first product is {images:right}
    """,
    )
    processed_1.rename(
        mapper=lambda x: x.replace(":", "_"), axis="columns", inplace=True
    )

    processed_2 = bottomwear.sem_join(
        topwear,
        """
     You will receive a description and an image of two products. Determine whether they are from the same brand.
     The description of the first product is as follows: {productDisplayName:left} {productDescriptors:left}.
     And the image of the first product is {images:left}.
     The description of the second product is as follows: {productDisplayName:right} {productDescriptors:right}.
     And the image of the first product is {images:right}
    """,
    )
    processed_2.rename(
        mapper=lambda x: x.replace(":", "_"), axis="columns", inplace=True
    )

    processed = processed_1.sem_join(
        processed_2,
        """
     You will receive a description and an image of two products. Determine whether they are from the same brand.
     The description of the first product is as follows: {productDisplayName_right:left} {productDescriptors_right:left}.
     And the image of the first product is {images_right:left}.
     The description of the second product is as follows: {productDisplayName_left:right} {productDescriptors_left:right}.
     And the image of the first product is {images_left:right}
    """,
    )

    processed["id"] = (
        processed["id_left:left"].astype("str")
        + "-"
        + processed["id_left:right"].astype("str")
        + "-"
        + processed["id_right:right"].astype("str")
        + "-"
        + processed["id_right:left"].astype("str")
    )
    return processed["id"]
