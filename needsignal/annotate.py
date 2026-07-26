from __future__ import annotations

import argparse
import textwrap
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

VALID_CONFIDENCE = {"high", "medium", "low"}


def ask_required(prompt: str, valid_values: set[str]) -> str:
    """Request a value until the user supplies an allowed response."""

    while True:
        value = input(prompt).strip().lower()

        if value == "q":
            return "q"

        if value in valid_values:
            return value

        print(
            "Choose one of: "
            + ", ".join(sorted(valid_values))
            + " — or enter q to quit."
        )


def ask_optional(prompt: str) -> str:
    """Request optional annotation text."""

    return input(prompt).strip()


def print_record(row: pd.Series, current: int, total: int) -> None:
    """Display one discussion in a readable terminal format."""

    divider = "=" * 78

    print("\n" + divider)
    print(f"RECORD {current} OF {total}")
    print(f"Annotation ID: {row.get('annotation_id', '')}")
    print(f"Issue: {row.get('issue_number', '')}")
    print(f"Title: {row.get('title', '')}")
    print(f"URL: {row.get('issue_url', '')}")
    print(
        f"Heuristic signals: "
        f"{row.get('matched_patterns', '')}"
    )
    print(divider)

    thread_text = str(row.get("thread_text", ""))

    print(
        textwrap.fill(
            thread_text,
            width=100,
            replace_whitespace=False,
            drop_whitespace=False,
        )
    )

    print(divider)


def annotate_dataset(
    input_path: Path,
    limit: int | None,
) -> None:
    """Annotate unlabeled discussions and save after every record."""

    dataframe = pd.read_csv(
        input_path,
        keep_default_na=False,
    )

    required_columns = {
        "annotation_id",
        "thread_text",
        "primary_label",
        "has_workaround",
        "user_goal",
        "obstacle",
        "workaround",
        "human_cost",
        "underlying_need",
        "evidence_quote",
        "confidence",
        "annotator_notes",
    }

    missing = required_columns.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"Annotation file is missing columns: {sorted(missing)}"
        )

    unlabeled_indices = dataframe.index[
        dataframe["primary_label"].astype(str).str.strip() == ""
    ].tolist()

    if limit is not None:
        unlabeled_indices = unlabeled_indices[:limit]

    if not unlabeled_indices:
        print("No unlabeled records remain.")
        return

    completed = 0
    total = len(unlabeled_indices)

    for position, dataframe_index in enumerate(
        unlabeled_indices,
        start=1,
    ):
        row = dataframe.loc[dataframe_index]

        print_record(
            row=row,
            current=position,
            total=total,
        )

        label = ask_required(
            "\nPrimary label: ",
            VALID_LABELS,
        )

        if label == "q":
            print("\nProgress saved. Exiting annotation session.")
            break

        has_workaround = (
            "yes" if label == "workaround" else "no"
        )

        dataframe.at[
            dataframe_index,
            "primary_label",
        ] = label

        dataframe.at[
            dataframe_index,
            "has_workaround",
        ] = has_workaround

        if has_workaround == "yes":
            dataframe.at[
                dataframe_index,
                "user_goal",
            ] = ask_optional("User goal: ")

            dataframe.at[
                dataframe_index,
                "obstacle",
            ] = ask_optional("Obstacle: ")

            dataframe.at[
                dataframe_index,
                "workaround",
            ] = ask_optional("Workaround behavior: ")

            dataframe.at[
                dataframe_index,
                "human_cost",
            ] = ask_optional("Human cost: ")

            dataframe.at[
                dataframe_index,
                "underlying_need",
            ] = ask_optional("Underlying need: ")

            dataframe.at[
                dataframe_index,
                "evidence_quote",
            ] = ask_optional("Evidence quote: ")

        confidence = ask_required(
            "Confidence [high/medium/low]: ",
            VALID_CONFIDENCE,
        )

        if confidence == "q":
            confidence = ""

        dataframe.at[
            dataframe_index,
            "confidence",
        ] = confidence

        dataframe.at[
            dataframe_index,
            "annotator_notes",
        ] = ask_optional("Optional notes: ")

        dataframe.to_csv(
            input_path,
            index=False,
        )

        completed += 1
        print(
            f"\nSaved {row['annotation_id']} "
            f"({completed} completed this session)."
        )

    print(f"\nSession complete. Records labeled: {completed}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually annotate NeedSignal discussions."
    )

    parser.add_argument(
        "--input",
        default=(
            "data/annotations/"
            "needsignal_annotation.csv"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to label this session.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {input_path}"
        )

    annotate_dataset(
        input_path=input_path,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()