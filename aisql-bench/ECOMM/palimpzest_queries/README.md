SemBench's own Palimpzest programs for the ECOMM queries, copied verbatim from the SemBench repository
(`files/ecomm/queries/dialects/palimpzest/qN.py`, github.com/sembench/sembench): q1–q13 (SemBench wrote no
Palimpzest program for q14). Each defines `run(pz_config, data_dir)`; `data_dir` is this suite's
`data/sf_500` (styles_details.parquet, image_mapping.parquet, images/). Run by ../../palimpzest_exec.py in
Palimpzest's interpreter; the runner reads the output's `product_id` column (SemBench's runner renames it to `id`).
