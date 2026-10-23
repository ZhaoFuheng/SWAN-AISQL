"""ECOMM q14 for Palimpzest -- OUR program (SemBench wrote none for q14), the same shape as the SWAN and LOTUS
copies: products under 130 x images of white socks, joined on "the image fits the description", each product
keeping its best-matching image (SWAN: ai_score + QUALIFY row_number() = 1; LOTUS: sem_topk K=1 per product);
the output is the matched image ids."""
import os

import pandas as pd
import palimpzest as pz


def run(pz_config, data_dir: str):
    styles = pd.read_parquet(os.path.join(data_dir, "styles_details.parquet")).rename(
        columns={"id": "prod_id"}  # prevent naming conflict with internal Palimpzest 'id' column
    )
    styles["price_num"] = pd.to_numeric(styles["price"], errors="coerce")
    styles = styles[styles["price_num"] < 130][["prod_id", "productDisplayName", "productDescriptors"]].reset_index(drop=True)
    styles = pz.MemoryDataset(id="styles_q14", vals=styles)

    images = pd.read_parquet(os.path.join(data_dir, "image_mapping.parquet")).rename(columns={"id": "image_id"})
    images = images[["image_id", "filename"]].reset_index(drop=True)
    images = pz.MemoryDataset(id="images_q14", vals=images)
    images = images.add_columns(
        udf=lambda row: {"image_file_path": os.path.join(data_dir, "images", str(row["filename"]))},
        cols=[{"name": "image_file_path", "type": pz.ImageFilepath, "description": ""}],  # empty: it influences the LLM
    )

    # Semantic filter: white socks
    socks = images.sem_filter(
        "The image depicts white socks. If there are multiple products in the picture, always refer to the most prominent one.",
        depends_on=["image_file_path"],
    )

    # Semantic join: the image fits the product description
    pairs = styles.sem_join(
        socks,
        "The image fits the description of the product.",
        depends_on=["productDisplayName", "productDescriptors", "image_file_path"],
    )

    # Semantic score of each surviving pair, then the best image per product
    scored = pairs.sem_add_columns(
        [{"name": "fit_score", "type": int,
          "desc": "How well the image fits the product description, as an integer from 1 (poor fit) to 5 (perfect fit). Only output the integer."}],
        depends_on=["productDisplayName", "productDescriptors", "image_file_path"],
    )
    scored = scored.project(["prod_id", "image_id", "fit_score"])
    output = scored.run(pz_config)

    df = output.to_df()
    if len(df):
        df["fit_score"] = pd.to_numeric(df["fit_score"], errors="coerce").fillna(0)
        df = df.sort_values(["prod_id", "fit_score"], ascending=[True, False]).drop_duplicates("prod_id")
        df = pd.DataFrame({"product_id": df["image_id"].astype(str)})
    else:
        df = pd.DataFrame({"product_id": []})
    return {"results": df, "execution_stats": output.execution_stats}
