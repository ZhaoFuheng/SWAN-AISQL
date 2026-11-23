SemBench's own Palimpzest programs for the ECOMM queries, copied verbatim from the SemBench repository
(`files/ecomm/queries/dialects/palimpzest/qN.py`, github.com/sembench/sembench): q1–q13, all but q12 unchanged
(SemBench wrote no Palimpzest program for q14). Each defines `run(pz_config, data_dir)`; `data_dir` is this suite's
`data/sf_500` (styles_details.parquet, image_mapping.parquet, images/). Run by ../../palimpzest_exec.py in
Palimpzest's interpreter; the runner reads the output's `product_id` column (SemBench's runner renames it to `id`).

q12 is changed. SemBench's version asks one generated string field (`product_id`) for the whole JSON: Palimpzest
left that field empty for 47 of the 63 products and returned the other 16 as parsed objects, so the query scored
0.0 whatever the answers. Ours generates `brand` and `category` in the same single call, with SemBench's own
instructions, and assembles the JSON from them and the product id, as the SWAN-AISQL query does.
