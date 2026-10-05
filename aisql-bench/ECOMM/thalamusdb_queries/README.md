SemBench's own ThalamusDB formulations of the ECOMM queries, copied from the SemBench repository
(`files/ecomm/queries/dialects/thalamusdb/`, github.com/sembench/sembench): the five queries SemBench runs on
ThalamusDB (q1, q2, q7, q8, q9; the others need map, classify or ranking operators it does not have). One
edit: q8 reads the description length from `styles_details.description`, this suite's flattened column, in
place of SemBench's nested `productDescriptors.description.value`. Tables follow SemBench's ThalamusDB setup:
`styles_details` with `full_product_description` (title and description concatenated, single quotes removed)
and `image_mapping` with `local_image_path`.
