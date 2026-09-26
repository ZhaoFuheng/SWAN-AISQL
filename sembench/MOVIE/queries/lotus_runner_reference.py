"""
Created on May 28, 2025

@author: Jiale Lao

LOTUS system runner implementation based on generic_lotus_runner for movie use case.
"""

import pandas as pd
import time
from typing import Dict, Any, List
import lotus
from lotus.models import LM
from pathlib import Path
import re

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_lotus_runner.generic_lotus_runner import GenericLotusRunner

# Import additional modules for approximate policy
from lotus.models import SentenceTransformersRM
from lotus.types import CascadeArgs
from lotus.vector_store import FaissVS


class LotusRunner(GenericLotusRunner):
    """Runner for LOTUS system."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        """
        Initialize LOTUS runner.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
        """
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )

        # Initialize components for approximate policy
        if hasattr(self, "policy") and self.policy == "approximate":
            self.rm_text = SentenceTransformersRM(model="intfloat/e5-base-v2")
            self.vs = FaissVS()
            self.cascade_args = CascadeArgs(
                recall_target=0.8, precision_target=0.8
            )

    def _configure_lotus_for_join_type(self, join_type: str):
        """Configure LOTUS settings based on join type (text-only for movie sentiment comparisons)."""
        if hasattr(self, "policy") and self.policy == "approximate":
            if join_type == "text":
                lotus.settings.configure(
                    lm=self.lm, rm=self.rm_text, vs=self.vs
                )

    def _execute_q1(self) -> pd.DataFrame:
        """
        Execute Q1: Find clearly positive movie reviews.

        Returns:
            DataFrame with columns: reviewId
        """
        # Load reviews data
        reviews = self.load_data("Reviews.csv")

        # Semantic filter for clearly positive reviews
        filtered_reviews = reviews.sem_filter(
            'Determine if the following movie review is clearly positive. Review: "{reviewText}".'
        )

        # Check if we got any results
        if len(filtered_reviews) == 0:
            print("  Warning: No positive reviews found")
            return self._get_empty_results_dataframe(1)

        # Limit to 5 results
        top5_reviews = filtered_reviews.head(5)

        # Format output - evaluator only needs reviewId (first column)
        results = []
        for _, row in top5_reviews.iterrows():
            results.append({"reviewId": row["reviewId"]})

        result_df = pd.DataFrame(results)

        # Ensure columns exist even if empty
        if len(result_df) == 0:
            result_df = self._get_empty_results_dataframe(1)

        return result_df

    def _execute_q2(self) -> pd.DataFrame:
        """
        Execute Q2: Find five positive reviews for movie "taken_3"

        Returns:
            DataFrame with columns: reviewId
        """
        # Load reviews data
        reviews = self.load_data("Reviews.csv")
        reviews = reviews[reviews["id"] == "taken_3"]

        # Semantic filter for clearly positive reviews
        filtered_reviews = reviews.sem_filter(
            'Determine if the following movie review is clearly positive. Review: "{reviewText}".'
        )

        # Check if we got any results
        if len(filtered_reviews) == 0:
            print("  Warning: No positive reviews found")
            return self._get_empty_results_dataframe(1)

        # Limit to 5 results
        top5_reviews = filtered_reviews.head(5)

        # Format output - evaluator only needs reviewId (first column)
        results = []
        for _, row in top5_reviews.iterrows():
            results.append({"reviewId": row["reviewId"]})

        result_df = pd.DataFrame(results)

        # Ensure columns exist even if empty
        if len(result_df) == 0:
            result_df = self._get_empty_results_dataframe(1)

        return result_df

    def _execute_q3(self) -> pd.DataFrame:
        """
        Execute Q3: Count of positive reviews for a "bad" movie (taken_3).

        Returns:
            DataFrame with columns: positive_review_cnt
        """
        # Load reviews data
        reviews = self.load_data("Reviews.csv")

        # Filter for taken_3 movie
        filtered_reviews = reviews[reviews["id"] == "taken_3"]

        # Semantic filter for positive reviews
        positive_reviews = filtered_reviews.sem_filter(
            "Determine if the following review is clearly positive. Review: {reviewText}"
        )

        # Get count
        positive_review_cnt = positive_reviews.shape[0]

        # Return as DataFrame
        result_df = pd.DataFrame({"positive_review_cnt": [positive_review_cnt]})

        return result_df

    def _execute_q4(self) -> pd.DataFrame:
        """
        Execute Q4: Positivity ratio (average of 0/1) for a "bad" movie (taken_3).

        Returns:
            DataFrame with columns: positivity_ratio
        """
        # Load reviews data
        reviews = self.load_data("Reviews.csv")

        # Filter for taken_3 movie
        taken_reviews = reviews[reviews["id"] == "taken_3"]

        if len(taken_reviews) == 0:
            return pd.DataFrame({"positivity_ratio": [0.0]})

        # Use sem_filter to select positive reviews
        positive_reviews = taken_reviews.sem_filter(
            "Determine if the following review is clearly positive. Review: {reviewText}."
        )

        # Compute positivity ratio
        positivity_ratio = len(positive_reviews) / len(taken_reviews)

        # Return as DataFrame
        return pd.DataFrame({"positivity_ratio": [positivity_ratio]})

    def _execute_q5(self) -> pd.DataFrame:
        """
        Execute Q5: Find pairs of reviews with same sentiment for movie with id 'ant_man_and_the_wasp_quantumania' using semantic join.

        Returns:
            DataFrame with columns: id, reviewId, reviewId2
        """
        # Configure for text-only comparison
        self._configure_lotus_for_join_type("text")

        # Load reviews data and filter for specific movie
        reviews = self.load_data("Reviews.csv")
        reviews = reviews[reviews["id"] == "ant_man_and_the_wasp_quantumania"]

        # Reset index for approximate policy
        if hasattr(self, "policy") and self.policy == "approximate":
            reviews = reviews.reset_index(drop=True)

        # Check if we have reviews for this movie
        if len(reviews) == 0:
            print(
                "  Warning: No reviews found for movie 'ant_man_and_the_wasp_quantumania'"
            )
            return self._get_empty_results_dataframe(5)

        # Semantic self-join for same sentiment within specific movie
        join_instruction = 'These two movie reviews express the same sentiment - either both are positive or both are negative. Review 1: "{reviewText:left}" Review 2: "{reviewText:right}"'

        if hasattr(self, "policy") and self.policy == "approximate":
            joined_df = reviews.sem_join(
                reviews,
                join_instruction=join_instruction,
                cascade_args=self.cascade_args,
            )
        else:
            joined_df = reviews.sem_join(
                reviews, join_instruction=join_instruction
            )

        # Filter out self-matches (same reviewId)
        joined_df = joined_df[
            joined_df["reviewId:left"] != joined_df["reviewId:right"]
        ]

        # Check if we got any results
        if len(joined_df) == 0:
            print("  Warning: No matching review pairs found")
            return self._get_empty_results_dataframe(5)

        # Limit to 10 results
        final_result = joined_df.head(10)

        # Select and rename relevant columns to match expected format (evaluator needs id, reviewId, reviewId2)
        result_df = final_result[
            ["id:left", "reviewId:left", "reviewId:right"]
        ].copy()
        result_df.columns = ["id", "reviewId", "reviewId2"]

        # Ensure columns exist even if empty
        if len(result_df) == 0:
            result_df = self._get_empty_results_dataframe(5)

        return result_df

    def _execute_q6(self) -> pd.DataFrame:
        """
        Execute Q6: Find pairs of reviews with opposite sentiment for movie with id 'ant_man_and_the_wasp_quantumania' using semantic join.

        Returns:
            DataFrame with columns: id, reviewId, reviewId2
        """
        # Configure for text-only comparison
        self._configure_lotus_for_join_type("text")

        # Load reviews data and filter for specific movie
        reviews = self.load_data("Reviews.csv")
        reviews = reviews[reviews["id"] == "ant_man_and_the_wasp_quantumania"]

        # Reset index for approximate policy
        if hasattr(self, "policy") and self.policy == "approximate":
            reviews = reviews.reset_index(drop=True)

        # Check if we have reviews for this movie
        if len(reviews) == 0:
            print(
                "  Warning: No reviews found for movie 'ant_man_and_the_wasp_quantumania'"
            )
            return self._get_empty_results_dataframe(6)

        # Semantic self-join for opposite sentiment within specific movie
        join_instruction = 'These two movie reviews express opposite sentiments - one is positive and the other is negative. Review 1: "{reviewText:left}" Review 2: "{reviewText:right}"'

        if hasattr(self, "policy") and self.policy == "approximate":
            joined_df = reviews.sem_join(
                reviews,
                join_instruction=join_instruction,
                cascade_args=self.cascade_args,
            )
        else:
            joined_df = reviews.sem_join(
                reviews, join_instruction=join_instruction
            )

        # Filter out self-matches (same reviewId)
        joined_df = joined_df[
            joined_df["reviewId:left"] != joined_df["reviewId:right"]
        ]

        # Check if we got any results
        if len(joined_df) == 0:
            print(
                "  Warning: No matching opposite sentiment review pairs found"
            )
            return self._get_empty_results_dataframe(6)

        # Limit to 10 results
        final_result = joined_df.head(10)

        # Select and rename relevant columns to match expected format (evaluator needs id, reviewId, reviewId2)
        result_df = final_result[
            ["id:left", "reviewId:left", "reviewId:right"]
        ].copy()
        result_df.columns = ["id", "reviewId", "reviewId2"]

        # Ensure columns exist even if empty
        if len(result_df) == 0:
            result_df = self._get_empty_results_dataframe(6)

        return result_df

    def _execute_q7(self) -> pd.DataFrame:
        """
        Execute Q7: All Pairs of reviews that express the *opposite* sentiment for movie with id 'ant_man_and_the_wasp_quantumania'.

        Returns:
            DataFrame with columns: id, reviewId1, reviewId2
        """
        # Configure for text-only comparison
        self._configure_lotus_for_join_type("text")

        # Load reviews data and filter for specific movie
        reviews = self.load_data("Reviews.csv")
        reviews = reviews[reviews["id"] == "ant_man_and_the_wasp_quantumania"]

        # Reset index for approximate policy
        if hasattr(self, "policy") and self.policy == "approximate":
            reviews = reviews.reset_index(drop=True)

        # Check if we have reviews for this movie
        if len(reviews) == 0:
            print(
                "  Warning: No reviews found for movie 'ant_man_and_the_wasp_quantumania'"
            )
            return self._get_empty_results_dataframe(6)

        # Semantic self-join for opposite sentiment within specific movie
        join_instruction = 'These two movie reviews express opposite sentiments - one is positive and the other is negative. Review 1: "{reviewText:left}" Review 2: "{reviewText:right}"'

        if hasattr(self, "policy") and self.policy == "approximate":
            joined_df = reviews.sem_join(
                reviews,
                join_instruction=join_instruction,
                cascade_args=self.cascade_args,
            )
        else:
            joined_df = reviews.sem_join(
                reviews, join_instruction=join_instruction
            )

        # Filter out self-matches (same reviewId)
        joined_df = joined_df[
            joined_df["reviewId:left"] != joined_df["reviewId:right"]
        ]

        # Check if we got any results
        if len(joined_df) == 0:
            print(
                "  Warning: No matching opposite sentiment review pairs found"
            )
            return self._get_empty_results_dataframe(6)

        # Select and rename relevant columns to match expected format (evaluator needs id, reviewId, reviewId2)
        result_df = joined_df[
            ["id:left", "reviewId:left", "reviewId:right"]
        ].copy()
        result_df.columns = ["id", "reviewId", "reviewId2"]

        # Ensure columns exist even if empty
        if len(result_df) == 0:
            result_df = self._get_empty_results_dataframe(6)

        return result_df

    def _execute_q8(self) -> pd.DataFrame:
        """
        Execute Q8: Calculate the number of positive and negative reviews for movie "taken_3".

        Returns:
            DataFrame with columns: scoreSentiment, count
        """
        # Load reviews data
        reviews = self.load_data("Reviews.csv")

        # Filter for taken_3 movie
        filtered_reviews = reviews[reviews["id"] == "taken_3"]

        # Semantic map to classify sentiment
        sentiment_reviews = filtered_reviews.sem_map(
            "Classify the sentiment of this review as either 'POSITIVE' or 'NEGATIVE'. "
            "Only output the exact word 'POSITIVE' or 'NEGATIVE' with no additional text. "
            "Review: {reviewText}"
        )

        # Count sentiment occurrences
        sentiment_counts = (
            sentiment_reviews["_map"].value_counts().reset_index()
        )
        sentiment_counts.columns = ["scoreSentiment", "count"]

        # Ensure we have both sentiment types in results (even if count is 0)
        expected_sentiments = ["POSITIVE", "NEGATIVE"]
        for sentiment in expected_sentiments:
            if sentiment not in sentiment_counts["scoreSentiment"].values:
                new_row = pd.DataFrame(
                    {"scoreSentiment": [sentiment], "count": [0]}
                )
                sentiment_counts = pd.concat(
                    [sentiment_counts, new_row], ignore_index=True
                )

        # Sort by sentiment for consistent output
        sentiment_counts = sentiment_counts.sort_values(
            "scoreSentiment"
        ).reset_index(drop=True)

        return sentiment_counts

    def _execute_q9(self) -> pd.DataFrame:
        """
        Execute Q9: Score from 1 to 5 how much did the reviewer like the movie based on the movie reviews for movie 'ant_man_and_the_wasp_quantumania'.

        Returns:
            DataFrame with columns: reviewId, reviewScore
        """
        # Load reviews data
        reviews = self.load_data("Reviews.csv")

        # Filter for ant_man_and_the_wasp_quantumania movie
        filtered_reviews = reviews[
            reviews["id"] == "ant_man_and_the_wasp_quantumania"
        ]

        if len(filtered_reviews) == 0:
            print(
                "  Warning: No reviews found for movie 'ant_man_and_the_wasp_quantumania'"
            )
            return pd.DataFrame(columns=["reviewId", "reviewScore"])

        # Define the scoring prompt based on BigQuery implementation
        scoring_prompt = """Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review: {reviewText}

Only provide the score number (1-5) with no other comments."""

        if self.ranking == "map":
            # Use sem_map for scoring - returns scores directly
            scored_reviews = filtered_reviews.sem_map(scoring_prompt)

            # Extract scores from the _map column
            results = []
            for _, row in scored_reviews.iterrows():
                score = row["_map"]
                # Ensure score is numeric and within range 1-5
                try:
                    numeric_score = float(score)
                    if 1 <= numeric_score <= 5:
                        results.append(
                            {
                                "reviewId": row["reviewId"],
                                "reviewScore": numeric_score,
                            }
                        )
                    else:
                        # Default to 3 if score is out of range
                        results.append(
                            {"reviewId": row["reviewId"], "reviewScore": 3.0}
                        )
                except (ValueError, TypeError):
                    # Default to 3 if score is not numeric
                    results.append(
                        {"reviewId": row["reviewId"], "reviewScore": 3.0}
                    )

        else:  # topk method
            # Use sem_topk for ranking and manually assign scores based on rank
            # We'll rank all reviews and assign scores based on their rank

            # Use exact policy (default LOTUS configuration)
            ranked_reviews, stats = filtered_reviews.sem_topk(
                """Which review shows the most positive sentiment about the movie? Review: {reviewText}""",
                K=len(filtered_reviews),  # Get all reviews ranked
                method="quick",
                return_stats=True,
            )

            results = []
            for idx, (_, row) in enumerate(ranked_reviews.iterrows()):
                # Assign scores based on rank position
                # Top ranked (most positive) gets 5, bottom gets 1
                # Distribute scores evenly across the range
                rank_position = idx
                total_reviews = len(ranked_reviews)

                if total_reviews == 1:
                    assigned_score = 3.0  # Neutral for single review
                else:
                    # Map rank position to score 1-5
                    # rank 0 (best) -> 5, rank (total-1) (worst) -> 1
                    score_range = 4.0  # 5 - 1
                    normalized_position = rank_position / (
                        total_reviews - 1
                    )  # 0 to 1
                    assigned_score = 5.0 - (normalized_position * score_range)
                    assigned_score = round(
                        assigned_score, 1
                    )  # Round to 1 decimal place

                results.append(
                    {"reviewId": row["reviewId"], "reviewScore": assigned_score}
                )

        result_df = pd.DataFrame(results)
        return result_df

    def _execute_q10(self) -> pd.DataFrame:
        """
        Execute Q10: Rank the movies based on movie reviews. For each movie, score every review of it from 1 to 5, then calculate the average score of these reviews for each movie.

        Returns:
            DataFrame with columns: movieId, movieScore
        """
        # Load reviews data
        reviews = self.load_data("Reviews.csv")

        if len(reviews) == 0:
            print("  Warning: No reviews found")
            return pd.DataFrame(columns=["movieId", "movieScore"])

        # Define the scoring prompt based on BigQuery implementation
        scoring_prompt = """Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review: {reviewText}

Only provide the score number (1-5) with no other comments."""

        # Use sem_map for scoring all reviews - simpler and more reliable
        scored_reviews = reviews.sem_map(scoring_prompt)

        # Extract scores from the _map column and group by movie to calculate averages
        movie_scores = []
        grouped_scored = scored_reviews.groupby("id")

        for movie_id, movie_scored_reviews in grouped_scored:
            valid_scores = []
            for _, row in movie_scored_reviews.iterrows():
                score = row["_map"]
                try:
                    numeric_score = float(score)
                    if 1 <= numeric_score <= 5:
                        valid_scores.append(numeric_score)
                    else:
                        valid_scores.append(3.0)  # Default to neutral
                except (ValueError, TypeError):
                    valid_scores.append(3.0)  # Default to neutral

            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
            else:
                avg_score = 3.0  # Default to neutral

            movie_scores.append(
                {"movieId": movie_id, "movieScore": round(avg_score, 2)}
            )

        result_df = pd.DataFrame(movie_scores)
        return result_df
