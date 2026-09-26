-- SemBench ECOMM sf_500 -> SWAN tables. Run with cwd = sembench/ECOMM (relative data paths; ai_image()
-- resolves image paths relative to the duckdb process cwd). Nested parquet structs are flattened here so the
-- AI queries read plain columns; the ground truth runs on the RAW parquets separately (gen_ground_truth.py).
CREATE TABLE styles_details AS
SELECT id,
       productDisplayName                        AS title,
       productDescriptors.description.value      AS description,
       brandName,
       articleType.typeName                      AS articleType,
       subCategory.typeName                      AS subCategory,
       masterCategory.typeName                   AS masterCategory,
       baseColour, colour1, colour2, price, gender, usage, season
FROM read_parquet('data/sf_500/styles_details.parquet');

CREATE TABLE images AS
SELECT CAST(id AS BIGINT)            AS id,
       filename,
       'data/sf_500/images/' || filename AS filepath
FROM read_parquet('data/sf_500/image_mapping.parquet');
