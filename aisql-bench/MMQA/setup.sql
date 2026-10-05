-- SemBench MMQA (sf200 = 200 rows, RANDOM_SEED=42, matches the paper). Load the sf_200 tables.
-- Run from swan-ai-sql/aisql-bench/MMQA:  build/reldebug/duckdb mmqa.db -init setup.sql
-- Then each Q*.sql references these tables.
CREATE OR REPLACE TABLE ben_piazza AS
  SELECT * FROM read_csv('files/mmqa/data/sf_200/ben_piazza.csv', header=true);              -- Year,Title,Role,Notes (19)
CREATE OR REPLACE TABLE ben_piazza_text_data AS
  SELECT * FROM read_csv('files/mmqa/data/sf_200/ben_piazza_text_data.csv', header=true);    -- row_id,title,url,id,text (200)
CREATE OR REPLACE TABLE lizzy_caplan_text_data AS
  SELECT * FROM read_csv('files/mmqa/data/sf_200/lizzy_caplan_text_data.csv', header=true);  -- row_id,title,url,text (200)
CREATE OR REPLACE TABLE tampa_international_airport AS
  SELECT * FROM read_csv('files/mmqa/data/sf_200/tampa_international_airport.csv', header=true); -- row_id,Airlines,Destinations,Airport (200)
CREATE OR REPLACE TABLE ap_warrior AS
  SELECT * FROM read_csv('files/mmqa/data/sf_200/ap_warrior.csv', header=true);               -- ID,Finish,Race,Distance,Track,Condition (13)
CREATE OR REPLACE TABLE images AS
  SELECT * FROM read_csv('files/mmqa/data/sf_200/thalamusdb_images.csv', header=true);         -- row_id,image_filename,image_filepath (200)
-- Q2/Q7 use images via ai_image(image_filepath) -> SWAN sends the image to the multimodal model (vision).
