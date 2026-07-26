from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VALID_LABELS = {
    "workaround",
    "feature_request",
    "bug_report",
    "user_confusion",
    "general_complaint",
    "not_actionable",
}

VALID_CONFIDENCE = {
    "high",
    "medium",
    "low",
}

WORKAROUND_FIELDS = [
    "user_goal",
    "obstacle",
    "workaround",
    "human_cost",
    "underlying_need",
    "evidence_quote",
]


def clean(value: object) -> str:
    """Convert a dataframe value into clean lowercase text."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def validate_annotations(
    dataframe: pd.DataFrame,
) -> list[dict[str, str]]:
    """Return all annotation problems found in the dataset."""

    errors: list[dict[str, str]] = []

    required_columns = {
        "annotation_id",
        "primary_label",
        "has_workaround",
        "confidence",
        *WORKAROUND_FIELDS,
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    duplicate_mask = dataframe["annotation_id"].duplicated(
        keep=False
    )

    for annotation_id in dataframe.loc[
        duplicate_mask,
        "annotation_id",
    ]:
        errors.append(
            {
                "annotation_id": clean(annotation_id),
                "problem": "Duplicate annotation_id",
            }
        )

    for _, row in dataframe.iterrows():
        annotation_id = clean(row["annotation_id"])
        label = clean(row["primary_label"]).lower()

        # Ignore records that have not been labeled yet.
        if not label:
            continue

        if label not in VALID_LABELS:
            errors.append(
                {
                    "annotation_id": annotation_id,
                    "problem": (
                        f"Invalid primary label: {label}"
                    ),
                }
            )
            continue

        has_workaround = clean(
            row["has_workaround"]
        ).lower()

        confidence = clean(
            row["confidence"]
        ).lower()

        if confidence not in VALID_CONFIDENCE:
            errors.append(
                {
                    "annotation_id": annotation_id,
                    "problem": (
                        "Confidence must be high, medium, "
                        f"or low. Found: {confidence or 'blank'}"
                    ),
                }
            )

        if label == "workaround":
            if has_workaround != "yes":
                errors.append(
                    {
                        "annotation_id": annotation_id,
                        "problem": (
                            "Workaround label must have "
                            "has_workaround=yes"
                        ),
                    }
                )

            for field in WORKAROUND_FIELDS:
                value = clean(row[field])

                if not value:
                    errors.append(
                        {
                            "annotation_id": annotation_id,
                            "problem": (
                                f"Workaround record is missing "
                                f"{field}"
                            ),
                        }
                    )

        else:
            if has_workaround != "no":
                errors.append(
                    {
                        "annotation_id": annotation_id,
                        "problem": (
                            "Non-workaround label must have "
                            "has_workaround=no"
                        ),
                    }
                )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate NeedSignal human annotations."
    )

    parser.add_argument(
        "--input",
        default=(
            "data/annotations/"
            "needsignal_annotation_labeled.csv"
        ),
    )

    parser.add_argument(
        "--errors-output",
        default=(
            "data/annotations/"
            "annotation_validation_errors.csv"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {input_path}"
        )

    dataframe = pd.read_csv(
        input_path,
        keep_default_na=False,
    )

    errors = validate_annotations(dataframe)

    labeled_mask = (
        dataframe["primary_label"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    labeled = dataframe.loc[labeled_mask]

    print()
    print(f"Total discussions: {len(dataframe)}")
    print(f"Labeled discussions: {len(labeled)}")
    print(
        f"Unlabeled discussions: "
        f"{len(dataframe) - len(labeled)}"
    )

    if len(labeled):
        print()
        print("Label counts:")
        print(
            labeled["primary_label"]
            .value_counts()
            .to_string()
        )

    print()

    if errors:
        errors_dataframe = pd.DataFrame(errors)

        errors_path = Path(args.errors_output)
        errors_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        errors_dataframe.to_csv(
            errors_path,
            index=False,
        )

        print(
            f"Validation found {len(errors)} problem(s)."
        )
        print(
            f"Review them here: {errors_path}"
        )
        print()
        print(errors_dataframe.to_string(index=False))

        raise SystemExit(1)

    print("Validation passed. No annotation problems found.")


if __name__ == "__main__":
    main()