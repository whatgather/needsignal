from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("reports/false_negative_analysis.csv")
DEFAULT_OUTPUT = Path("reports/false_negative_audit.csv")


EVIDENCE_OPTIONS = {
    "e": {
        "evidence_status": "explicit",
        "audit_outcome": "model_failure",
    },
    "i": {
        "evidence_status": "implicit_but_supported",
        "audit_outcome": "contextual_inference_failure",
    },
    "n": {
        "evidence_status": "not_present",
        "audit_outcome": "annotation_error",
    },
}


def clean_text(value: object) -> str:
    """Convert missing values into clean strings."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def display_record(
    row: pd.Series,
    position: int,
    total: int,
) -> None:
    """Display one false-negative record for review."""

    divider = "=" * 90

    print("\n" + divider)
    print(f"FALSE NEGATIVE {position} OF {total}")
    print(f"Annotation ID: {clean_text(row.get('annotation_id'))}")
    print(f"Issue number: {clean_text(row.get('issue_number'))}")
    print(f"Title: {clean_text(row.get('title'))}")
    print(f"Model probability: {clean_text(row.get('probability'))}")
    print(divider)

    structured_fields = [
        ("USER GOAL", "user_goal"),
        ("OBSTACLE", "obstacle"),
        ("LABELED WORKAROUND", "workaround"),
        ("HUMAN COST", "human_cost"),
        ("UNDERLYING NEED", "underlying_need"),
        ("EVIDENCE QUOTE", "evidence_quote"),
    ]

    for heading, column in structured_fields:
        value = clean_text(row.get(column))

        if value:
            print(f"\n{heading}:")
            print(
                textwrap.fill(
                    value,
                    width=100,
                    replace_whitespace=False,
                )
            )

    discussion = clean_text(row.get("thread_text"))

    if not discussion:
        discussion = clean_text(row.get("analysis_text"))

    if discussion:
        print("\nFULL DISCUSSION:")
        print(
            textwrap.fill(
                discussion,
                width=100,
                replace_whitespace=False,
                drop_whitespace=False,
            )
        )

    print("\n" + divider)
    print("e = Explicit workaround behaviour is directly stated")
    print("i = Behaviour is indirect, but clearly supported by context")
    print("n = No compensating behaviour appears in the discussion")
    print("q = Save and quit")


def request_evidence_status() -> str:
    """Ask for a valid evidence classification."""

    while True:
        answer = input(
            "\nEvidence classification [e/i/n/q]: "
        ).strip().lower()

        if answer in {"e", "i", "n", "q"}:
            return answer

        print("Enter e, i, n, or q.")


def audit_false_negatives(
    input_path: Path,
    output_path: Path,
    limit: int | None,
) -> None:
    """Audit false negatives and save after every decision."""

    if output_path.exists():
        dataframe = pd.read_csv(
            output_path,
            keep_default_na=False,
        )
        print(f"Continuing existing audit: {output_path}")
    else:
        if not input_path.exists():
            raise FileNotFoundError(
                f"False-negative file not found: {input_path}"
            )

        dataframe = pd.read_csv(
            input_path,
            keep_default_na=False,
        )

    required_columns = {"annotation_id"}

    missing = required_columns.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"Input file is missing columns: {sorted(missing)}\n"
            f"Available columns: {dataframe.columns.tolist()}"
        )

    audit_columns = [
        "evidence_status",
        "audit_outcome",
        "audit_notes",
    ]

    for column in audit_columns:
        if column not in dataframe.columns:
            dataframe[column] = ""

    unreviewed_indices = dataframe.index[
        dataframe["evidence_status"]
        .astype(str)
        .str.strip()
        .eq("")
    ].tolist()

    if limit is not None:
        unreviewed_indices = unreviewed_indices[:limit]

    if not unreviewed_indices:
        print("All false negatives have already been audited.")
        print_summary(dataframe)
        return

    total = len(unreviewed_indices)
    reviewed_this_session = 0

    for position, dataframe_index in enumerate(
        unreviewed_indices,
        start=1,
    ):
        row = dataframe.loc[dataframe_index]

        display_record(
            row=row,
            position=position,
            total=total,
        )

        answer = request_evidence_status()

        if answer == "q":
            print("\nProgress saved. Ending audit session.")
            break

        decision = EVIDENCE_OPTIONS[answer]

        dataframe.at[
            dataframe_index,
            "evidence_status",
        ] = decision["evidence_status"]

        dataframe.at[
            dataframe_index,
            "audit_outcome",
        ] = decision["audit_outcome"]

        dataframe.at[
            dataframe_index,
            "audit_notes",
        ] = input("Optional audit notes: ").strip()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        reviewed_this_session += 1

        print(
            f"\nSaved {row['annotation_id']} as "
            f"{decision['evidence_status']}."
        )

    print(
        f"\nRecords reviewed this session: "
        f"{reviewed_this_session}"
    )

    print_summary(dataframe)


def print_summary(dataframe: pd.DataFrame) -> None:
    """Print the current audit findings."""

    reviewed = dataframe[
        dataframe["evidence_status"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    print("\nAUDIT SUMMARY")
    print(f"Total false negatives: {len(dataframe)}")
    print(f"Reviewed: {len(reviewed)}")
    print(f"Remaining: {len(dataframe) - len(reviewed)}")

    if reviewed.empty:
        return

    print("\nEvidence status:")
    print(
        reviewed["evidence_status"]
        .value_counts()
        .to_string()
    )

    print("\nLikely cause:")
    print(
        reviewed["audit_outcome"]
        .value_counts()
        .to_string()
    )

    annotation_errors = (
        reviewed["audit_outcome"] == "annotation_error"
    ).sum()

    if annotation_errors:
        print(
            "\nImportant: records classified as annotation_error "
            "should be relabeled before retraining."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit NeedSignal false-negative predictions."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    audit_false_negatives(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()