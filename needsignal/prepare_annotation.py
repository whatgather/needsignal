from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


WORKAROUND_PATTERNS = {
    "manual_process": (
        r"\b(manually|manual process|by hand|copy[- ]?paste|"
        r"copying and pasting)\b"
    ),
    "forced_action": (
        r"\b(have to|had to|need to|forced to|must first)\b"
    ),
    "alternative_method": (
        r"\b(instead|alternative|another tool|different tool|"
        r"external tool)\b"
    ),
    "explicit_workaround": (
        r"\b(workaround|work around|temporary fix|hacky solution|hack)\b"
    ),
    "repetition": (
        r"\b(every time|again and again|repeatedly|repeat this|"
        r"redo|re-enter)\b"
    ),
    "data_transfer": (
        r"\b(export|import|spreadsheet|csv|download and upload|"
        r"move the data)\b"
    ),
    "custom_code": (
        r"\b(custom script|write a script|small script|"
        r"custom code|unofficial script)\b"
    ),
    "only_method": (
        r"\b(the only way|only way I can|currently I have to|"
        r"the best I can do)\b"
    ),
}


def normalize_text(value: object) -> str:
    """Convert missing values to empty text and clean whitespace."""

    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def combine_comments(series: pd.Series) -> str:
    """Join all usable comments from one issue discussion."""

    comments = [
        normalize_text(comment)
        for comment in series
        if normalize_text(comment)
    ]

    return "\n\n--- COMMENT ---\n\n".join(comments)


def calculate_heuristic_score(text: str) -> tuple[int, str]:
    """
    Identify language that may indicate a workaround.

    This does not determine the final label. It only helps us sample
    discussions that are more likely to contain useful evidence.
    """

    matched_patterns: list[str] = []

    for pattern_name, pattern in WORKAROUND_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            matched_patterns.append(pattern_name)

    return len(matched_patterns), " | ".join(matched_patterns)


def build_thread_text(row: pd.Series) -> str:
    """Create one readable document from an issue and its comments."""

    sections = [
        f"TITLE:\n{normalize_text(row.get('title'))}",
        f"ISSUE DESCRIPTION:\n{normalize_text(row.get('body'))}",
    ]

    comments_text = normalize_text(row.get("comments_text"))

    if comments_text:
        sections.append(f"DISCUSSION:\n{comments_text}")

    return "\n\n".join(sections)


def sample_threads(
    threads: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Select a mixture of likely signals and ordinary discussions.

    Including ordinary discussions is important because the final
    model must learn what is not a workaround.
    """

    if sample_size < 1:
        raise ValueError("Sample size must be at least 1.")

    sample_size = min(sample_size, len(threads))

    signal_pool = threads[
        threads["heuristic_score"] > 0
    ].copy()

    comparison_pool = threads[
        threads["heuristic_score"] == 0
    ].copy()

    signal_target = min(
        len(signal_pool),
        round(sample_size * 0.7),
    )

    signal_sample = signal_pool.sample(
        n=signal_target,
        random_state=random_state,
    ) if signal_target else signal_pool.head(0)

    remaining = sample_size - len(signal_sample)

    comparison_target = min(
        len(comparison_pool),
        remaining,
    )

    comparison_sample = comparison_pool.sample(
        n=comparison_target,
        random_state=random_state,
    ) if comparison_target else comparison_pool.head(0)

    selected = pd.concat(
        [signal_sample, comparison_sample],
        ignore_index=True,
    )

    remaining = sample_size - len(selected)

    if remaining > 0:
        unused = threads[
            ~threads["issue_number"].isin(
                selected["issue_number"]
            )
        ]

        extra = unused.sample(
            n=min(remaining, len(unused)),
            random_state=random_state,
        )

        selected = pd.concat(
            [selected, extra],
            ignore_index=True,
        )

    return selected.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)


def prepare_annotation_dataset(
    issues: pd.DataFrame,
    comments: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    """Create the first human-annotation dataset."""

    issue_columns = {
        "repository",
        "issue_number",
        "title",
        "body",
        "issue_url",
    }

    missing_issue_columns = issue_columns.difference(
        issues.columns
    )

    if missing_issue_columns:
        raise ValueError(
            "Issues file is missing columns: "
            f"{sorted(missing_issue_columns)}"
        )

    comment_columns = {
        "issue_number",
        "comment_body",
    }

    missing_comment_columns = comment_columns.difference(
        comments.columns
    )

    if missing_comment_columns:
        raise ValueError(
            "Comments file is missing columns: "
            f"{sorted(missing_comment_columns)}"
        )

    if "position_in_thread" in comments.columns:
        comments = comments.sort_values(
            ["issue_number", "position_in_thread"]
        )

    comment_threads = (
        comments.groupby("issue_number")["comment_body"]
        .apply(combine_comments)
        .rename("comments_text")
        .reset_index()
    )

    threads = issues.merge(
        comment_threads,
        on="issue_number",
        how="left",
    )

    threads["comments_text"] = (
        threads["comments_text"].fillna("")
    )

    threads["thread_text"] = threads.apply(
        build_thread_text,
        axis=1,
    )

    scores = threads["thread_text"].apply(
        calculate_heuristic_score
    )

    threads["heuristic_score"] = scores.apply(
        lambda result: result[0]
    )

    threads["matched_patterns"] = scores.apply(
        lambda result: result[1]
    )

    selected = sample_threads(
        threads=threads,
        sample_size=sample_size,
    )

    selected.insert(
        0,
        "annotation_id",
        [
            f"NS-{number:04d}"
            for number in range(1, len(selected) + 1)
        ],
    )

    # These columns are intentionally blank for human annotation.
    selected["primary_label"] = ""
    selected["has_workaround"] = ""
    selected["user_goal"] = ""
    selected["obstacle"] = ""
    selected["workaround"] = ""
    selected["human_cost"] = ""
    selected["underlying_need"] = ""
    selected["evidence_quote"] = ""
    selected["confidence"] = ""
    selected["annotator_notes"] = ""

    final_columns = [
        "annotation_id",
        "repository",
        "issue_number",
        "issue_url",
        "title",
        "thread_text",
        "comments_count",
        "heuristic_score",
        "matched_patterns",
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
    ]

    existing_columns = [
        column
        for column in final_columns
        if column in selected.columns
    ]

    return selected[existing_columns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a human-annotation dataset for NeedSignal."
        )
    )

    parser.add_argument("--issues", required=True)
    parser.add_argument("--comments", required=True)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument(
        "--output",
        default="data/annotations/needsignal_annotation.csv",
    )

    args = parser.parse_args()

    issues_path = Path(args.issues)
    comments_path = Path(args.comments)

    if not issues_path.exists():
        raise FileNotFoundError(
            f"Issues file not found: {issues_path}"
        )

    if not comments_path.exists():
        raise FileNotFoundError(
            f"Comments file not found: {comments_path}"
        )

    issues = pd.read_csv(issues_path)
    comments = pd.read_csv(comments_path)

    annotation_dataset = prepare_annotation_dataset(
        issues=issues,
        comments=comments,
        sample_size=args.sample_size,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotation_dataset.to_csv(
        output_path,
        index=False,
    )

    likely_signals = (
        annotation_dataset["heuristic_score"] > 0
    ).sum()

    print()
    print(
        f"Discussions prepared: "
        f"{len(annotation_dataset)}"
    )
    print(
        f"Possible workaround signals: "
        f"{likely_signals}"
    )
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()