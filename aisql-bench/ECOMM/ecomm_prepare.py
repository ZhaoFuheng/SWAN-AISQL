#!/usr/bin/env python3
"""Build the full parquets + the sf_500 sample from the extracted metadata (no images yet).

Faithful port of SemBench src/scenario/ecomm/download.py: same read_csv/read_json SQL, same sample SQL
(reservoir seed 12345600, 43 forced query-solution rows), same joins. Emits data/sf_500/*.parquet and
data/sf_500/image_files.txt (tar member paths of the 500 sampled images, for the second selective pass).
"""
import os
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "source", "1", "fashion-dataset")
FULL = os.path.join(HERE, "data", "full")
OUT = os.path.join(HERE, "data", "sf_500")
os.makedirs(FULL, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
SEED = 12345600
SF = 500

con = duckdb.connect()
con.execute("SET threads = 2;")
con.execute("SET preserve_insertion_order = false;")

p = lambda *a: os.path.join(*a).replace("'", "''")

if not os.path.exists(p(FULL, "styles.parquet")):
    con.execute(f"""COPY (SELECT * FROM read_csv('{p(SRC,'styles.csv')}', header=true, delim=',', quote='"',
        ignore_errors=true)) TO '{p(FULL,'styles.parquet')}' (FORMAT PARQUET)""")
    print("full styles.parquet done", flush=True)
if not os.path.exists(p(FULL, "image_mapping.parquet")):
    con.execute(f"""COPY (SELECT split_part(filename, '.', 1) as id, filename, link
        FROM read_csv('{p(SRC,'images.csv')}', header=true))
        TO '{p(FULL,'image_mapping.parquet')}' (FORMAT PARQUET)""")
    print("full image_mapping.parquet done", flush=True)
if not os.path.exists(p(FULL, "styles_details.parquet")):
    con.execute(f"""COPY (SELECT unnest(data) FROM read_json('{p(SRC,'styles','*.json')}', union_by_name=true))
        TO '{p(FULL,'styles_details.parquet')}' (FORMAT PARQUET)""")
    print("full styles_details.parquet done", flush=True)

# --- the sample, verbatim from SemBench's _create_sample (num_extra_rows=43) ---
con.execute(f"""
    COPY (
        SELECT * FROM read_parquet('{p(FULL,'styles.parquet')}') USING SAMPLE {SF - 43} (reservoir, {SEED})
        UNION SELECT * FROM read_parquet('{p(FULL,'styles.parquet')}') WHERE id IN (5299, 5300, 5301, 1623, 1624, 5303, 5314)
        UNION SELECT * FROM read_parquet('{p(FULL,'styles.parquet')}') WHERE id IN (10037, 10102, 3312, 41825, 3462)
        UNION SELECT * FROM read_parquet('{p(FULL,'styles.parquet')}') WHERE id IN (3351, 30292, 10689, 8419)
        UNION SELECT * FROM read_parquet('{p(FULL,'styles.parquet')}') WHERE id IN (12799, 2048, 2606, 2607, 3479, 4038, 4800, 4805, 4817, 2045, 43047, 4811)
        UNION SELECT * FROM read_parquet('{p(FULL,'styles.parquet')}') WHERE id IN (6241, 1891, 53126, 1563, 15779, 47525)
        UNION SELECT * FROM read_parquet('{p(FULL,'styles.parquet')}') WHERE id IN (6100, 7935, 10579)
        UNION SELECT * FROM read_parquet('{p(FULL,'styles.parquet')}') WHERE id IN (8103, 13112, 8402, 3470)
        UNION SELECT * FROM read_parquet('{p(FULL,'styles.parquet')}') WHERE id IN (43047, 12799, 4811)
        UNION SELECT * FROM read_parquet('{p(FULL,'styles.parquet')}') WHERE id IN (18345, 29202)
    ) TO '{p(OUT,'styles.parquet')}' (FORMAT PARQUET)""")
con.execute(f"""COPY (SELECT styles_details.* FROM read_parquet('{p(OUT,'styles.parquet')}') AS styles
    JOIN read_parquet('{p(FULL,'styles_details.parquet')}') AS styles_details ON styles_details.id = styles.id)
    TO '{p(OUT,'styles_details.parquet')}' (FORMAT PARQUET)""")
con.execute(f"""COPY (SELECT image_mapping.* FROM read_parquet('{p(OUT,'styles.parquet')}') AS styles
    JOIN read_parquet('{p(FULL,'image_mapping.parquet')}') AS image_mapping ON image_mapping.id = styles.id)
    TO '{p(OUT,'image_mapping.parquet')}' (FORMAT PARQUET)""")

files = [r[0] for r in con.execute(
    f"SELECT filename FROM read_parquet('{p(OUT,'image_mapping.parquet')}')").fetchall()]
with open(os.path.join(OUT, "image_files.txt"), "w") as f:
    f.write("\n".join(f"1/fashion-dataset/images/{fn}" for fn in files) + "\n")
n = con.execute(f"SELECT count(*) FROM read_parquet('{p(OUT,'styles.parquet')}')").fetchone()[0]
print(f"sf_500 built: {n} styles rows, {len(files)} images listed", flush=True)
