from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PREDICTIONS_PATH = Path("reports/baseline_predictions.csv")
ANNOTATIONS_PATH = Path(
    "data/annotations/needsignal_annotation_labeled.csv"
)

FALSE_NEGATIVES_PATH = Path(
    "reports/false_negative_analysis.csv"
)

PATTERN_COUNTS_PATH = Path(
    "reports/false_negative_pattern_counts.csv"
)

REPORT_PATH = Path(
    "reports/error_analysis.md"
)


COLUMN_ALIASES = {
    "actual": [
        "actual",
        "actual_label",
    ],
    "predicted": [
        "predicted",
        "model_prediction",
    ],
    "probability": [
        "probability",
        "workaround_probability",
    ],
}


BEHAVIORAL_PATTERNS = {
    "constraint_language": (
        r"\b(have to|had to|cannot|can't|unable|"
        r"not supported|only way|doesn't allow)\b"
    ),
    "manual_intervention": (
        r"\b(manually|by hand|copy|paste|edit|modify|"
        r"export|import|download|upload)\b"
    ),
    "alternative_method": (
        r"\b(instead|alternative|another tool|different tool|"
        r"workaround|custom script|custom code|hack)\b"
    ),
    "repetition_or_retry": (
        r"\b(every time|repeatedly|again|retry|reconnect|"
        r"restart|refresh|redo)\b"
    ),
    "multi_step_sequence": (
        r"\b(first|then|before|after|next|until|followed by)\b"
    ),
    "successful_compensation": (
        r"\b(this works|worked for me|fixes it|solved it|"
        r"able to continue|temporary fix)\b"
    ),
}


def clean_headers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove spaces and hidden characters from column names."""

    dataframe = dataframe.copy()

    dataframe.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in dataframe.columns
    ]

    return dataframe


def apply_column_aliases(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Rename known prediction columns to standard names."""

    dataframe = dataframe.copy()
    rename_map: dict[str, str] = {}

    for standard_name, possible_names in COLUMN_ALIASES.items():
        if standard_name in dataframe.columns:
            continue

        for possible_name in possible_names:
            if possible_name in dataframe.columns:
                rename_map[possible_name] = standard_name
                break

    return dataframe.rename(columns=rename_map)


def normalize_label(value: object) -> str:
    """Normalize binary workaround labels."""

    label = str(value).strip().lower()

    if label in {
        "workaround",
        "yes",
        "true",
        "1",
        "positive",
    }:
        return "workaround"

    if label in {
        "no_workaround",
        "no workaround",
        "no",
        "false",
        "0",
        "negative",
    }:
        return "no_workaround"

    return label


def combine_text(row: pd.Series) -> str:
    """Combine all useful evidence for pattern analysis."""

    fields = [
        "title",
        "thread_text",
        "user_goal",
        "obstacle",
        "workaround",
        "human_cost",
        "underlying_need",
        "evidence_quote",
    ]

    parts = []

    for field in fields:
        value = row.get(field, "")

        if pd.notna(value) and str(value).strip():
            parts.append(str(value).strip())

    return "\n".join(parts)


def identify_patterns(text: str) -> list[str]:
    """Find behavioral language contained in one discussion."""

    matches = []

    for pattern_name, pattern in BEHAVIORAL_PATTERNS.items():
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            matches.append(pattern_name)

    return matches


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {PREDICTIONS_PATH}"
        )

    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing annotations file: {ANNOTATIONS_PATH}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        keep_default_na=False,
    )

    annotations = pd.read_csv(
        ANNOTATIONS_PATH,
        keep_default_na=False,
    )

    predictions = clean_headers(predictions)
    annotations = clean_headers(annotations)
    predictions = apply_column_aliases(predictions)

    required_prediction_columns = {
        "annotation_id",
        "actual",
        "predicted",
        "probability",
    }

    missing = required_prediction_columns.difference(
        predictions.columns
    )

    if missing:
        raise ValueError(
            f"Predictions file is missing: {sorted(missing)}\n"
            f"Available columns: {predictions.columns.tolist()}"
        )

    predictions["actual"] = predictions["actual"].apply(
        normalize_label
    )

    predictions["predicted"] = predictions["predicted"].apply(
        normalize_label
    )

    predictions["probability"] = pd.to_numeric(
        predictions["probability"],
        errors="coerce",
    )

    annotation_fields = [
        "annotation_id",
        "thread_text",
        "user_goal",
        "obstacle",
        "workaround",
        "human_cost",
        "underlying_need",
        "evidence_quote",
        "confidence",
    ]

    available_annotation_fields = [
        column
        for column in annotation_fields
        if column in annotations.columns
    ]

    fields_to_merge = [
        column
        for column in available_annotation_fields
        if (
            column == "annotation_id"
            or column not in predictions.columns
        )
    ]

    combined = predictions.merge(
        annotations[fields_to_merge],
        on="annotation_id",
        how="left",
    )

    combined["analysis_text"] = combined.apply(
        combine_text,
        axis=1,
    )

    combined["behavioral_cues"] = combined[
        "analysis_text"
    ].apply(identify_patterns)

    combined["behavioral_cue_count"] = combined[
        "behavioral_cues"
    ].apply(len)

    combined["behavioral_cues"] = combined[
        "behavioral_cues"
    ].apply(lambda values: " | ".join(values))

    false_negatives = combined[
        (combined["actual"] == "workaround")
        & (combined["predicted"] == "no_workaround")
    ].copy()

    false_positives = combined[
        (combined["actual"] == "no_workaround")
        & (combined["predicted"] == "workaround")
    ].copy()

    false_negatives = false_negatives.sort_values(
        by=[
            "behavioral_cue_count",
            "probability",
        ],
        ascending=[
            False,
            True,
        ],
    )

    output_columns = [
        "annotation_id",
        "issue_number",
        "title",
        "probability",
        "behavioral_cue_count",
        "behavioral_cues",
        "user_goal",
        "obstacle",
        "workaround",
        "human_cost",
        "underlying_need",
        "evidence_quote",
        "thread_text",
    ]

    available_output_columns = [
        column
        for column in output_columns
        if column in false_negatives.columns
    ]

    FALSE_NEGATIVES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    false_negatives[
        available_output_columns
    ].to_csv(
        FALSE_NEGATIVES_PATH,
        index=False,
    )

    exploded_patterns = (
        false_negatives["behavioral_cues"]
        .str.split(r" \| ")
        .explode()
    )

    exploded_patterns = exploded_patterns[
        exploded_patterns.astype(str).str.strip() != ""
    ]

    pattern_counts = (
        exploded_patterns.value_counts()
        .rename_axis("behavioral_pattern")
        .reset_index(name="false_negative_count")
    )

    pattern_counts.to_csv(
        PATTERN_COUNTS_PATH,
        index=False,
    )

    evaluated = len(combined)

    correct = int(
        (
            combined["actual"]
            == combined["predicted"]
        ).sum()
    )

    report_lines = [
        "# NeedSignal Baseline Error Analysis",
        "",
        "## Baseline summary",
        "",
        f"- Evaluated records: {evaluated}",
        f"- Correct predictions: {correct}",
        f"- False negatives: {len(false_negatives)}",
        f"- False positives: {len(false_positives)}",
        "",
        "## Behavioral cues inside missed workarounds",
        "",
    ]

    if pattern_counts.empty:
        report_lines.append(
            "No predefined behavioral cues were detected."
        )
    else:
        report_lines.extend(
            [
                "| Behavioral cue | Missed cases |",
                "|---|---:|",
            ]
        )

        for row in pattern_counts.itertuples(index=False):
            report_lines.append(
                f"| {row.behavioral_pattern} | "
                f"{row.false_negative_count} |"
            )

    report_lines.extend(
        [
            "",
            "## Missed workaround examples",
            "",
        ]
    )

    for row in false_negatives.itertuples(index=False):
        title = getattr(row, "title", "")
        annotation_id = getattr(row, "annotation_id", "")
        probability = getattr(row, "probability", "")
        cues = getattr(row, "behavioral_cues", "")

        report_lines.extend(
            [
                f"### {annotation_id}: {title}",
                "",
                f"- Model probability: {probability}",
                f"- Behavioral cues: {cues or 'none detected'}",
                "",
            ]
        )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("ERROR ANALYSIS COMPLETE")
    print(f"Evaluated records: {evaluated}")
    print(f"False negatives: {len(false_negatives)}")
    print(f"False positives: {len(false_positives)}")
    print()
    print(f"Saved: {FALSE_NEGATIVES_PATH}")
    print(f"Saved: {PATTERN_COUNTS_PATH}")
    print(f"Saved: {REPORT_PATH}")

    if not pattern_counts.empty:
        print()
        print("MOST COMMON CUES IN MISSED WORKAROUNDS")
        print(pattern_counts.to_string(index=False))


if __name__ == "__main__":
    main()