"""SemBench's own Palimpzest programs for the MOVIE queries, copied from the SemBench repository
(`src/scenario/movie/runner/palimpzest_runner/palimpzest_runner.py`, github.com/sembench/sembench), one
function per query: qN(pz_config, data_dir) -> the program's output. `data_dir` holds SemBench's
Reviews.csv / Movies.csv (this suite's data/sf_2000). Run by ../palimpzest_exec.py in Palimpzest's interpreter.
"""
import os

import palimpzest as pz
import pandas as pd
from palimpzest.core.elements.groupbysig import GroupBySig

RUBRIC = """Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review: {reviewText}

Only provide the score number (1-5) with no other comments."""


def load_data(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, name))


def q1(cfg, data_dir):
    reviews = pz.MemoryDataset(id="reviews", vals=load_data(data_dir, "Reviews.csv"))
    reviews = reviews.sem_filter("Determine if the following movie review is clearly positive.", depends_on=["reviewText"])
    reviews = reviews.project(["reviewId"])
    reviews = reviews.limit(5)
    return reviews.run(cfg)


def q2(cfg, data_dir):
    reviews = pz.MemoryDataset(id="reviews", vals=load_data(data_dir, "Reviews.csv").rename(columns={"id": "movieId"}))
    reviews = reviews.filter(lambda r: r["movieId"] == "taken_3")
    reviews = reviews.sem_filter("Determine if the following movie review is clearly positive.", depends_on=["reviewText"])
    reviews = reviews.project(["reviewId"])
    reviews = reviews.limit(5)
    return reviews.run(cfg)


def q3(cfg, data_dir):
    reviews = pz.MemoryDataset(id="reviews", vals=load_data(data_dir, "Reviews.csv").rename(columns={"id": "movieId"}))
    reviews = reviews.filter(lambda r: r["movieId"] == "taken_3")
    reviews = reviews.sem_filter("Determine if the following movie review is clearly positive.", depends_on=["reviewText"])
    reviews = reviews.count()
    return reviews.run(cfg)


def q4(cfg, data_dir):
    reviews = pz.MemoryDataset(id="reviews", vals=load_data(data_dir, "Reviews.csv").rename(columns={"id": "movieId"}))
    reviews = reviews.filter(lambda r: r["movieId"] == "taken_3")
    reviews = reviews.sem_add_columns(
        [{"name": "positivity", "type": int,
          "desc": "Return 1 if the following review is positive, and 0 if the review is not positive. Only output a single numeric value (1 or 0) with no additional commentary"}],
        depends_on=["reviewText"])
    reviews = reviews.project(["positivity"])
    reviews = reviews.average()
    return reviews.run(cfg)


def _pairs(cfg, data_dir, condition, limit):
    reviews_df = load_data(data_dir, "Reviews.csv").rename(columns={"id": "movieId"})
    input_df1 = reviews_df[reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"]
    input_df2 = reviews_df[reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"]
    input_df2 = input_df2.rename(columns={col: f"{col}_right" for col in input_df2.columns})
    input1 = pz.MemoryDataset(id="input1", vals=input_df1)
    input2 = pz.MemoryDataset(id="input2", vals=input_df2)
    input3 = input1.sem_join(input2, condition=condition, depends_on=["reviewText", "reviewText_right"])
    input3 = input3.project(["movieId", "reviewId", "reviewId_right"])
    if limit is not None:
        input3 = input3.limit(limit)
    return input3.run(cfg)


def q5(cfg, data_dir):
    return _pairs(cfg, data_dir, "These two movie reviews express the same sentiment - either both are positive or both are negative.", 10)


def q6(cfg, data_dir):
    return _pairs(cfg, data_dir, "These two movie reviews express opposite sentiments - one is positive and the other is negative.", 10)


def q7(cfg, data_dir):
    return _pairs(cfg, data_dir, "These two movie reviews express opposite sentiments - one is positive and the other is negative.", None)


def q8(cfg, data_dir):
    reviews = pz.MemoryDataset(id="reviews", vals=load_data(data_dir, "Reviews.csv").rename(columns={"id": "movieId"}))
    reviews = reviews.filter(lambda r: r["movieId"] == "taken_3")
    reviews = reviews.sem_add_columns(
        [{"name": "sentiment", "type": str,
          "desc": "Return POSITIVE if the following review is positive, and NEGATIVE if the review is not positive. Only output POSITIVE or NEGATIVE with no additional commentary"}],
        depends_on=["reviewText"])
    reviews = reviews.project(["sentiment"])
    reviews = reviews.groupby(GroupBySig(group_by_fields=["sentiment"], agg_funcs=["count"], agg_fields=["sentiment"]))
    return reviews.run(cfg)


def q9(cfg, data_dir):
    reviews_df = load_data(data_dir, "Reviews.csv").rename(columns={"id": "movieId"})
    filtered = reviews_df[reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"]
    reviews = pz.MemoryDataset(id="reviews", vals=filtered)
    reviews = reviews.sem_add_columns([{"name": "reviewScore", "type": int, "desc": RUBRIC}], depends_on=["reviewText"])
    reviews = reviews.project(["reviewId", "reviewScore"])
    return reviews.run(cfg)


def q10(cfg, data_dir):
    reviews = pz.MemoryDataset(id="reviews", vals=load_data(data_dir, "Reviews.csv").rename(columns={"id": "movieId"}))
    reviews = reviews.sem_add_columns([{"name": "reviewScore", "type": int, "desc": RUBRIC}], depends_on=["reviewText"])
    reviews = reviews.project(["movieId", "reviewScore"])
    reviews = reviews.groupby(GroupBySig(group_by_fields=["movieId"], agg_funcs=["average"], agg_fields=["reviewScore"]))
    return reviews.run(cfg)
