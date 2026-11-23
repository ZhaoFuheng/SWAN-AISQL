import json
import os
import pandas as pd
import palimpzest as pz


def add_image_data(
    styles_details: pz.MemoryDataset, data_dir: str
) -> pz.MemoryDataset:
    styles_details = styles_details.add_columns(
        udf=lambda row: {
            "image_file_path": os.path.join(
                data_dir, "images", str(row["prod_id"]) + ".jpg"
            )
        },
        cols=[
            {
                "name": "image_file_path",
                "type": pz.ImageFilepath,
                "description": "",  # leave empty because this influences the LLMs decision
            }
        ],
    )
    return styles_details


def run(pz_config, data_dir: str):
    # Load data
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    ).rename(
        columns={"id": "prod_id"}
    )  # prevent naming conflict with internal Palimpzest 'id' column

    styles_details = styles_details[styles_details.apply(lambda row: row['masterCategory']['typeName'] in ('Accessories', 'Apparel', 'Footwear'), axis=1)]
    
    styles_details = pz.MemoryDataset(id="styles_details", vals=styles_details)
    styles_details = add_image_data(styles_details, data_dir)

    # Filter data
    styles_details = styles_details.sem_filter(
        "Does the following description describe a product from either Adidas or Puma?",
        depends_on=[
            "productDisplayName",
            "productDescriptors",
        ],
    )

    # Project JSON. CHANGED from SemBench's program, which asked ONE generated string field (product_id) to
    # hold the whole JSON: Palimpzest left that field empty for 47 of the 63 products and returned the other
    # 16 as parsed objects, so the query scored 0.0 whatever the answers. Here the model generates the two
    # semantic values (SemBench's own instructions, one call per product) and the JSON is assembled from
    # them and the product id, as the SWAN-AISQL query assembles it.
    styles_details = styles_details.sem_add_columns(
        cols=[
            {
                "name": "brand",
                "type": str,
                "description": "Extract the brand name from the description and/or image. Use lower-case letters for the brand name.",
            },
            {
                "name": "category",
                "type": str,
                "description": "Classify the image into 'accessories', 'apparel', 'footwear'.",
            },
        ],
        desc="""
            You are given a product description and an image of the product.
            The product contains a fashion item (clothing, shoes, accessories, etc).
            There might be multiple fashion items in the image, especially when a model is presenting them.
            If this is the case, focus only on the primary fashion item and use the description to determine which item in the image is of interest.
        """,
        depends_on=["productDisplayName", "productDescriptors", "image_file_path"],
    )
    styles_details = styles_details.add_columns(
        udf=lambda row: {
            "product_id": json.dumps(
                {
                    "id": int(row["prod_id"]),
                    "brand": str(row["brand"] or "").strip().lower(),
                    "category": str(row["category"] or "").strip().lower(),
                },
                separators=(",", ":"),
            )
        },
        cols=[{"name": "product_id", "type": str, "description": ""}],
    )

    styles_details = styles_details.project(["product_id"])

    output = styles_details.run(pz_config)
    return output
