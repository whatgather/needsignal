from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline


def clean_text(value: object) -> str:
    """Convert missing values into usable text."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Calculate binary classification metrics."""

    return {
        "accuracy": round(
            float(accuracy_score(actual, predicted)),
            4,
        ),
        "precision": round(
            float(
                precision_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    actual,
                    predicted,
                    zero_division=0,
                )
            ),
            4,
        ),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ).tolist(),
    }


def build_pipeline() -> Pipeline:
    """Create the initial text-classification pipeline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    max_features=5000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def get_top_terms(
    fitted_pipeline: Pipeline,
    number_of_terms: int = 15,
) -> dict[str, list[dict[str, Any]]]:
    """
    Return terms most associated with each class.

    These are exploratory signals, not causal explanations.
    """

    vectorizer = fitted_pipeline.named_steps["tfidf"]
    classifier = fitted_pipeline.named_steps["classifier"]

    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    positive_indices = np.argsort(coefficients)[
        -number_of_terms:
    ][::-1]

    negative_indices = np.argsort(coefficients)[
        :number_of_terms
    ]

    workaround_terms = [
        {
            "term": str(feature_names[index]),
            "weight": round(
                float(coefficients[index]),
                4,
            ),
        }
        for index in positive_indices
    ]

    non_workaround_terms = [
        {
            "term": str(feature_names[index]),
            "weight": round(
                float(coefficients[index]),
                4,
            ),
        }
        for index in negative_indices
    ]

    return {
        "workaround_associated_terms": workaround_terms,
        "non_workaround_associated_terms": (
            non_workaround_terms
        ),
    }


def train_and_evaluate(
    dataframe: pd.DataFrame,
) -> tuple[
    Pipeline,
    pd.DataFrame,
    dict[str, Any],
]:
    """Train and evaluate the first NeedSignal detector."""

    required_columns = {
        "annotation_id",
        "thread_text",
        "has_workaround",
        "heuristic_score",
    }

    missing = required_columns.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    working = dataframe.copy()

    working["has_workaround"] = (
        working["has_workaround"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    working = working[
        working["has_workaround"].isin(
            {"yes", "no"}
        )
    ].copy()

    working["thread_text"] = (
        working["thread_text"]
        .apply(clean_text)
    )

    working = working[
        working["thread_text"].str.len() > 0
    ].copy()

    if len(working) < 10:
        raise ValueError(
            "At least 10 labeled discussions are required."
        )

    working["target"] = (
        working["has_workaround"]
        .map({"no": 0, "yes": 1})
        .astype(int)
    )

    class_counts = (
        working["target"]
        .value_counts()
        .sort_index()
    )

    if len(class_counts) < 2:
        raise ValueError(
            "The labeled dataset must contain both "
            "workaround and non-workaround examples."
        )

    minimum_class_count = int(
        class_counts.min()
    )

    if minimum_class_count < 2:
        raise ValueError(
            "Each class needs at least two examples. "
            f"Current counts: {class_counts.to_dict()}"
        )

    number_of_folds = min(
        5,
        minimum_class_count,
    )

    cross_validation = StratifiedKFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=42,
    )

    features = working["thread_text"]
    target = working["target"]

    pipeline = build_pipeline()

    probabilities = cross_val_predict(
        pipeline,
        features,
        target,
        cv=cross_validation,
        method="predict_proba",
    )[:, 1]

    model_predictions = (
        probabilities >= 0.5
    ).astype(int)

    heuristic_scores = pd.to_numeric(
        working["heuristic_score"],
        errors="coerce",
    ).fillna(0)

    heuristic_predictions = (
        heuristic_scores > 0
    ).astype(int)

    model_metrics = calculate_metrics(
        actual=target,
        predicted=model_predictions,
    )

    heuristic_metrics = calculate_metrics(
        actual=target,
        predicted=heuristic_predictions,
    )

    results = working.copy()

    results["actual_label"] = target.map(
        {
            0: "no_workaround",
            1: "workaround",
        }
    )

    results["heuristic_prediction"] = (
        heuristic_predictions.map(
            {
                0: "no_workaround",
                1: "workaround",
            }
        )
    )

    results["model_prediction"] = pd.Series(
        model_predictions,
        index=results.index,
    ).map(
        {
            0: "no_workaround",
            1: "workaround",
        }
    )

    results["workaround_probability"] = (
        probabilities.round(4)
    )

    results["model_correct"] = (
        model_predictions
        == target.to_numpy()
    )

    results["heuristic_correct"] = (
        heuristic_predictions.to_numpy()
        == target.to_numpy()
    )

    # Train the final model on every labeled example.
    pipeline.fit(
        features,
        target,
    )

    metrics = {
        "dataset": {
            "labeled_records": int(
                len(working)
            ),
            "workaround_records": int(
                (target == 1).sum()
            ),
            "non_workaround_records": int(
                (target == 0).sum()
            ),
            "cross_validation_folds": (
                number_of_folds
            ),
        },
        "tfidf_logistic_regression": (
            model_metrics
        ),
        "keyword_heuristic": (
            heuristic_metrics
        ),
        "interpretation_note": (
            "Results are preliminary because the "
            "human-labeled dataset is still small."
        ),
        "top_terms": get_top_terms(
            fitted_pipeline=pipeline,
        ),
    }

    return pipeline, results, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the first NeedSignal "
            "workaround detector."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "data/annotations/"
            "needsignal_annotation_labeled.csv"
        ),
    )

    parser.add_argument(
        "--model-output",
        default=(
            "models/"
            "workaround_detector.joblib"
        ),
    )

    parser.add_argument(
        "--predictions-output",
        default=(
            "reports/"
            "baseline_predictions.csv"
        ),
    )

    parser.add_argument(
        "--metrics-output",
        default=(
            "reports/"
            "baseline_metrics.json"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    dataframe = pd.read_csv(
        input_path,
        keep_default_na=False,
    )

    pipeline, predictions, metrics = (
        train_and_evaluate(dataframe)
    )

    model_path = Path(args.model_output)
    predictions_path = Path(
        args.predictions_output
    )
    metrics_path = Path(
        args.metrics_output
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    useful_prediction_columns = [
        "annotation_id",
        "repository",
        "issue_number",
        "title",
        "primary_label",
        "actual_label",
        "heuristic_score",
        "matched_patterns",
        "heuristic_prediction",
        "model_prediction",
        "workaround_probability",
        "heuristic_correct",
        "model_correct",
    ]

    existing_columns = [
        column
        for column in useful_prediction_columns
        if column in predictions.columns
    ]

    predictions[
        existing_columns
    ].to_csv(
        predictions_path,
        index=False,
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    model_metrics = metrics[
        "tfidf_logistic_regression"
    ]

    heuristic_metrics = metrics[
        "keyword_heuristic"
    ]

    print()
    print("NeedSignal baseline evaluation")
    print("=" * 40)

    print(
        "Labeled records: "
        f"{metrics['dataset']['labeled_records']}"
    )

    print(
        "Workarounds: "
        f"{metrics['dataset']['workaround_records']}"
    )

    print(
        "Non-workarounds: "
        f"{metrics['dataset']['non_workaround_records']}"
    )

    print()
    print("Keyword heuristic")
    print(
        f"Precision: {heuristic_metrics['precision']}"
    )
    print(
        f"Recall:    {heuristic_metrics['recall']}"
    )
    print(
        f"F1:        {heuristic_metrics['f1']}"
    )

    print()
    print("TF-IDF + Logistic Regression")
    print(
        f"Precision: {model_metrics['precision']}"
    )
    print(
        f"Recall:    {model_metrics['recall']}"
    )
    print(
        f"F1:        {model_metrics['f1']}"
    )

    print()
    print(f"Model saved to: {model_path}")
    print(
        f"Predictions saved to: "
        f"{predictions_path}"
    )
    print(
        f"Metrics saved to: {metrics_path}"
    )


if __name__ == "__main__":
    main()