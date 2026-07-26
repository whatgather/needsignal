from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/annotations/needsignal_annotation_labeled.csv"
)

OUTPUT_PATH = Path(
    "reports/opportunity_signals.csv"
)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def calculate_opportunity_score(row: pd.Series) -> float:
    """
    Produce a transparent first-pass opportunity score.

    This is not a machine-learning prediction. It ranks confirmed
    workarounds using observable evidence in the annotation dataset.
    """

    score = 0.0

    confidence_weights = {
        "high": 3.0,
        "medium": 2.0,
        "low": 1.0,
    }

    confidence = clean_text(
        row.get("confidence")
    ).lower()

    score += confidence_weights.get(confidence, 0.0)

    if clean_text(row.get("human_cost")):
        score += 2.0

    if clean_text(row.get("obstacle")):
        score += 1.5

    if clean_text(row.get("underlying_need")):
        score += 1.5

    if clean_text(row.get("evidence_quote")):
        score += 1.0

    comments_count = pd.to_numeric(
        row.get("comments_count", 0),
        errors="coerce",
    )

    if pd.notna(comments_count):
        score += min(float(comments_count), 10.0) * 0.1

    heuristic_score = pd.to_numeric(
        row.get("heuristic_score", 0),
        errors="coerce",
    )

    if pd.notna(heuristic_score):
        score += min(float(heuristic_score), 5.0) * 0.2

    return round(score, 2)


def determine_cost_type(row: pd.Series) -> str:
    text = clean_text(
        row.get("human_cost")
    ).lower()

    categories = []

    if any(
        word in text
        for word in [
            "time",
            "slow",
            "delay",
            "waiting",
        ]
    ):
        categories.append("time")

    if any(
        word in text
        for word in [
            "manual",
            "repeat",
            "effort",
            "extra step",
        ]
    ):
        categories.append("effort")

    if any(
        word in text
        for word in [
            "error",
            "risk",
            "lose",
            "failure",
            "break",
        ]
    ):
        categories.append("risk")

    if any(
        word in text
        for word in [
            "confusing",
            "frustrating",
            "difficult",
            "annoying",
        ]
    ):
        categories.append("friction")

    return " | ".join(categories) or "unspecified"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {INPUT_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_PATH,
        keep_default_na=False,
    )

    dataframe.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in dataframe.columns
    ]

    if "primary_label" not in dataframe.columns:
        raise ValueError(
            "The annotation file has no primary_label column."
        )

    opportunities = dataframe[
        dataframe["primary_label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("workaround")
    ].copy()

    if opportunities.empty:
        raise ValueError(
            "No confirmed workaround records were found."
        )

    opportunities["opportunity_score"] = (
        opportunities.apply(
            calculate_opportunity_score,
            axis=1,
        )
    )

    opportunities["cost_type"] = (
        opportunities.apply(
            determine_cost_type,
            axis=1,
        )
    )

    opportunities["opportunity_statement"] = (
        opportunities.apply(
            lambda row: (
                f"Users need a better way to "
                f"{clean_text(row.get('user_goal'))} "
                f"without having to "
                f"{clean_text(row.get('workaround'))}."
            ),
            axis=1,
        )
    )

    opportunities = opportunities.sort_values(
        "opportunity_score",
        ascending=False,
    )

    output_columns = [
        "annotation_id",
        "repository",
        "issue_number",
        "title",
        "user_goal",
        "obstacle",
        "workaround",
        "human_cost",
        "cost_type",
        "underlying_need",
        "opportunity_statement",
        "evidence_quote",
        "confidence",
        "opportunity_score",
        "issue_url",
    ]

    existing_columns = [
        column
        for column in output_columns
        if column in opportunities.columns
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    opportunities[
        existing_columns
    ].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Confirmed workarounds: "
        f"{len(opportunities)}"
    )
    print(f"Saved: {OUTPUT_PATH}")

    print()
    print("TOP OPPORTUNITY SIGNALS")

    preview_columns = [
        "annotation_id",
        "title",
        "cost_type",
        "opportunity_score",
    ]

    print(
        opportunities[
            preview_columns
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()