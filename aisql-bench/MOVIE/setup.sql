-- Build movie.db from the sf_2000 CSVs (run with cwd = aisql-bench/MOVIE)
CREATE TABLE movies  AS SELECT * FROM read_csv('data/sf_2000/Movies.csv');
CREATE TABLE reviews AS SELECT * FROM read_csv('data/sf_2000/Reviews.csv');
