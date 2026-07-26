from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import FeatureUnion, Pipeline


DATA_PATH = Path(
    "data/annotations/needsignal_annotation_labeled.csv"
)

BASELINE_METRICS_PATH = Path(
    "reports/baseline_metrics.json"
)

PREDICTIONS_PATH = Path(
    "reports/improved_predictions.csv"
)

ERRORS_PATH = Path(
    "reports/improved_errors.csv"
)

METRICS_PATH = Path(
    "reports/improved_metrics.json"
)

MODEL_PATH = Path(
    "models/improved_detector.joblib"
)


BEHAVIORAL_PATTERNS = {
    "constraint_language": (
        r"\b(have to|had to|cannot|can't|unable|"
        r"not supported|only way|doesn't allow|"
        r"won't let me|fails unless)\b"
    ),
    "manual_intervention": (
        r"\b(manually|by hand|copy|paste|edit|modify|"
        r"export|import|download|upload|re-enter)\b"
    ),
    "alternative_method": (
        r"\b(instead|alternative|another tool|different tool|"
        r"workaround|custom script|custom code|hack)\b"
    ),
    "repetition_or_retry": (
        r"\b(every time|repeatedly|again|retry|reconnect|"
        r"restart|refresh|redo|rerun)\b"
    ),
    "multi_step_sequence": (
        r"\b(first|then|before|after|next|until|"
        r"followed by)\b"
    ),
    "successful_compensation": (
        r"\b(this works|worked for me|fixes it|solved it|"
        r"able to continue|temporary fix|for now)\b"
    ),
    "avoidance_behavior": (
        r"\b(avoid|stopped using|don't use|disable|"
        r"turn off|skip)\b"
    ),
    "external_storage": (
        r"\b(spreadsheet|csv|excel|external database|"
        r"google sheets|local file)\b"
    ),
}


def clean_headers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean spaces and hidden characters from CSV headers."""

    dataframe = dataframe.copy()

    dataframe.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in dataframe.columns
    ]

    return dataframe


def normalize_text(value: object) -> str:
    """Convert a value into clean text."""

    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def build_text(row: pd.Series) -> str:
    """
    Create model input using only information available
    before human annotation.
    """

    title = normalize_text(row.get("title", ""))
    thread = normalize_text(row.get("thread_text", ""))

    parts = []

    if title:
        parts.append(f"TITLE: {title}")

    if thread:
        parts.append(thread)

    return "\n\n".join(parts)


def find_behavioral_cues(text: str) -> list[str]:
    """Return every behavioral pattern detected in the text."""

    detected = []

    for pattern_name, pattern in BEHAVIORAL_PATTERNS.items():
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            detected.append(pattern_name)

    return detected


def augment_text(text: str) -> str:
    """
    Add behavioral signal tokens to the text.

    Tokens are repeated to give the model a visible cue while still
    allowing it to learn from the original language.
    """

    cues = find_behavioral_cues(text)

    if not cues:
        return text

    cue_tokens = []

    for cue in cues:
        token = f"behavioral_cue_{cue}"

        # Repeat each cue twice so it has more influence.
        cue_tokens.extend([token, token])

    return text + "\n\n" + " ".join(cue_tokens)


def normalize_primary_label(value: object) -> int:
    """
    Convert the multiclass research label into a binary target.

    workaround = 1
    every other valid category = 0
    """

    label = str(value).strip().lower()

    return 1 if label == "workaround" else 0


def build_model() -> Pipeline:
    """Create the improved text-classification pipeline."""

    features = FeatureUnion(
        [
            (
                "word_features",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    sublinear_tf=True,
                    min_df=1,
                    max_features=12000,
                ),
            ),
            (
                "character_features",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=1,
                    max_features=12000,
                ),
            ),
        ]
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        C=1.5,
        max_iter=2000,
        random_state=42,
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


def classify_error(
    actual: str,
    predicted: str,
) -> str:
    """Name the type of binary classification mistake."""

    if (
        actual == "workaround"
        and predicted == "no_workaround"
    ):
        return "false_negative"

    if (
        actual == "no_workaround"
        and predicted == "workaround"
    ):
        return "false_positive"

    return ""


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found: {DATA_PATH}"
        )

    dataframe = pd.read_csv(
        DATA_PATH,
        keep_default_na=False,
    )

    dataframe = clean_headers(dataframe)

    required_columns = {
        "annotation_id",
        "primary_label",
    }

    missing = required_columns.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}\n"
            f"Available columns: {dataframe.columns.tolist()}"
        )

    labeled_mask = (
        dataframe["primary_label"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    dataframe = (
        dataframe.loc[labeled_mask]
        .copy()
        .reset_index(drop=True)
    )

    if dataframe.empty:
        raise ValueError(
            "The annotation dataset contains no labeled records."
        )

    dataframe["model_text"] = dataframe.apply(
        build_text,
        axis=1,
    )

    empty_text_mask = (
        dataframe["model_text"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_text_mask.any():
        print(
            f"Ignoring {int(empty_text_mask.sum())} records "
            "with no usable text."
        )

        dataframe = (
            dataframe.loc[~empty_text_mask]
            .copy()
            .reset_index(drop=True)
        )

    dataframe["behavioral_cues"] = dataframe[
        "model_text"
    ].apply(
        lambda text: " | ".join(
            find_behavioral_cues(text)
        )
    )

    dataframe["augmented_text"] = dataframe[
        "model_text"
    ].apply(augment_text)

    y = dataframe["primary_label"].apply(
        normalize_primary_label
    )

    positive_count = int(y.sum())
    negative_count = int(len(y) - positive_count)

    if positive_count < 2 or negative_count < 2:
        raise ValueError(
            "At least two workaround and two non-workaround "
            "records are required."
        )

    cv_folds = min(
        5,
        positive_count,
        negative_count,
    )

    cross_validator = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42,
    )

    model = build_model()

    probabilities = cross_val_predict(
        model,
        dataframe["augmented_text"],
        y,
        cv=cross_validator,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)

    actual_labels = np.where(
        y.to_numpy() == 1,
        "workaround",
        "no_workaround",
    )

    predicted_labels = np.where(
        predictions == 1,
        "workaround",
        "no_workaround",
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    accuracy = float(
        accuracy_score(
            y,
            predictions,
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            y,
            predictions,
        )
    )

    report_dictionary = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=[
            "no_workaround",
            "workaround",
        ],
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=[
            "no_workaround",
            "workaround",
        ],
        zero_division=0,
    )

    result_columns = [
        column
        for column in [
            "annotation_id",
            "repository",
            "issue_number",
            "title",
            "primary_label",
            "heuristic_score",
            "matched_patterns",
        ]
        if column in dataframe.columns
    ]

    results = dataframe[result_columns].copy()

    results["actual_label"] = actual_labels
    results["model_prediction"] = predicted_labels
    results["workaround_probability"] = probabilities
    results["behavioral_cues"] = dataframe[
        "behavioral_cues"
    ]

    results["model_correct"] = (
        results["actual_label"]
        == results["model_prediction"]
    )

    results["error_type"] = [
        classify_error(actual, predicted)
        for actual, predicted in zip(
            results["actual_label"],
            results["model_prediction"],
        )
    ]

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    errors = results[
        ~results["model_correct"]
    ].copy()

    errors = errors.sort_values(
        by=[
            "error_type",
            "workaround_probability",
        ],
        ascending=[
            True,
            True,
        ],
    )

    errors.to_csv(
        ERRORS_PATH,
        index=False,
    )

    baseline_comparison = {}

    if BASELINE_METRICS_PATH.exists():
        try:
            baseline_metrics = json.loads(
                BASELINE_METRICS_PATH.read_text(
                    encoding="utf-8"
                )
            )

            for key in [
                "accuracy",
                "false_negatives",
                "false_positives",
                "correct_predictions",
                "incorrect_predictions",
            ]:
                if key in baseline_metrics:
                    baseline_comparison[key] = (
                        baseline_metrics[key]
                    )

        except json.JSONDecodeError:
            baseline_comparison = {
                "warning": (
                    "Baseline metrics file could not be read."
                )
            }

    metrics = {
        "model_name": (
            "balanced_word_character_tfidf_logistic_regression"
        ),
        "evaluation_method": (
            f"{cv_folds}-fold stratified cross-validation"
        ),
        "threshold": threshold,
        "evaluated_records": int(len(dataframe)),
        "workaround_records": positive_count,
        "no_workaround_records": negative_count,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "correct_predictions": int(
            (predictions == y.to_numpy()).sum()
        ),
        "incorrect_predictions": int(
            (predictions != y.to_numpy()).sum()
        ),
        "confusion_matrix": {
            "true_negatives": true_negative,
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "true_positives": true_positive,
        },
        "workaround_precision": float(
            report_dictionary["workaround"]["precision"]
        ),
        "workaround_recall": float(
            report_dictionary["workaround"]["recall"]
        ),
        "workaround_f1": float(
            report_dictionary["workaround"]["f1-score"]
        ),
        "classification_report": report_dictionary,
        "baseline_comparison": baseline_comparison,
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Fit one final model on every labeled record for later use.
    model.fit(
        dataframe["augmented_text"],
        y,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "behavioral_patterns": BEHAVIORAL_PATTERNS,
            "model_name": metrics["model_name"],
        },
        MODEL_PATH,
    )

    print()
    print("IMPROVED MODEL COMPLETE")
    print(f"Records evaluated: {len(dataframe)}")
    print(f"Cross-validation folds: {cv_folds}")
    print(f"Accuracy: {accuracy:.3f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy:.3f}"
    )
    print(f"True positives: {true_positive}")
    print(f"True negatives: {true_negative}")
    print(f"False negatives: {false_negative}")
    print(f"False positives: {false_positive}")
    print(
        "Workaround precision: "
        f"{metrics['workaround_precision']:.3f}"
    )
    print(
        "Workaround recall: "
        f"{metrics['workaround_recall']:.3f}"
    )
    print(
        "Workaround F1: "
        f"{metrics['workaround_f1']:.3f}"
    )

    print("\nCLASSIFICATION REPORT")
    print(report_text)

    if baseline_comparison:
        print("\nBASELINE COMPARISON")

        baseline_accuracy = baseline_comparison.get(
            "accuracy"
        )

        baseline_false_negatives = (
            baseline_comparison.get(
                "false_negatives"
            )
        )

        baseline_false_positives = (
            baseline_comparison.get(
                "false_positives"
            )
        )

        if baseline_accuracy is not None:
            print(
                "Baseline accuracy: "
                f"{float(baseline_accuracy):.3f}"
            )

        if baseline_false_negatives is not None:
            print(
                "Baseline false negatives: "
                f"{baseline_false_negatives}"
            )

        if baseline_false_positives is not None:
            print(
                "Baseline false positives: "
                f"{baseline_false_positives}"
            )

    print()
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(f"Saved errors: {ERRORS_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()