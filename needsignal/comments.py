from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from .collect import create_headers, parse_next_link


def transform_comment(
    comment: dict[str, Any],
    repository: str,
    issue_number: int,
    position: int,
) -> dict[str, Any]:
    """Convert one GitHub comment into a clean dataset row."""

    body = comment.get("body") or ""
    user = comment.get("user") or {}

    return {
        "repository": repository,
        "issue_number": issue_number,
        "comment_id": comment.get("id"),
        "position_in_thread": position,
        "comment_body": body,
        "comment_length": len(body),
        "author_association": comment.get("author_association"),
        "user_type": user.get("type"),
        "created_at": comment.get("created_at"),
        "updated_at": comment.get("updated_at"),
        "comment_url": comment.get("html_url"),
    }


def fetch_issue_comments(
    owner: str,
    repo: str,
    issue_number: int,
    headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Fetch every comment belonging to one GitHub issue."""

    url: str | None = (
        f"https://api.github.com/repos/{owner}/{repo}"
        f"/issues/{issue_number}/comments"
    )

    params: dict[str, Any] | None = {
        "per_page": 100,
        "sort": "created",
        "direction": "asc",
    }

    rows: list[dict[str, Any]] = []
    position = 1

    while url:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            remaining = response.headers.get(
                "X-RateLimit-Remaining",
                "unknown",
            )

            raise RuntimeError(
                f"Comment request failed for issue "
                f"#{issue_number} with status "
                f"{response.status_code}. "
                f"Remaining requests: {remaining}. "
                f"Response: {response.text[:300]}"
            ) from error

        results = response.json()

        if not isinstance(results, list):
            raise RuntimeError(
                f"Unexpected response for issue #{issue_number}."
            )

        for comment in results:
            rows.append(
                transform_comment(
                    comment=comment,
                    repository=f"{owner}/{repo}",
                    issue_number=issue_number,
                    position=position,
                )
            )
            position += 1

        url = parse_next_link(response.headers.get("Link"))
        params = None

    return rows


def collect_comments(
    issues: pd.DataFrame,
    owner: str,
    repo: str,
    max_issues: int = 25,
) -> pd.DataFrame:
    """
    Collect comments from the richest issue discussions first.

    Starting with 25 issues helps avoid exhausting the API limit
    while we test the pipeline.
    """

    required_columns = {"issue_number", "comments_count"}

    missing = required_columns.difference(issues.columns)

    if missing:
        raise ValueError(
            f"Issues file is missing columns: {sorted(missing)}"
        )

    eligible = issues[
        issues["comments_count"].fillna(0).astype(int) > 0
    ].copy()

    eligible["comments_count"] = (
        eligible["comments_count"].fillna(0).astype(int)
    )

    eligible = eligible.sort_values(
        "comments_count",
        ascending=False,
    ).head(max_issues)

    load_dotenv()
    headers = create_headers()

    all_rows: list[dict[str, Any]] = []

    for index, issue in enumerate(
        eligible.itertuples(index=False),
        start=1,
    ):
        issue_number = int(issue.issue_number)

        print(
            f"[{index}/{len(eligible)}] "
            f"Collecting comments for issue #{issue_number}"
        )

        rows = fetch_issue_comments(
            owner=owner,
            repo=repo,
            issue_number=issue_number,
            headers=headers,
        )

        all_rows.extend(rows)

        # Be polite to the API.
        time.sleep(0.2)

    return pd.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect GitHub issue comments for NeedSignal."
    )

    parser.add_argument("--issues", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--max-issues", type=int, default=25)
    parser.add_argument("--output")

    args = parser.parse_args()

    issues_path = Path(args.issues)

    if not issues_path.exists():
        raise FileNotFoundError(
            f"Issues file not found: {issues_path}"
        )

    issues = pd.read_csv(issues_path)

    comments = collect_comments(
        issues=issues,
        owner=args.owner,
        repo=args.repo,
        max_issues=args.max_issues,
    )

    output_path = (
        Path(args.output)
        if args.output
        else Path("data/raw")
        / f"{args.owner}_{args.repo}_comments.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comments.to_csv(output_path, index=False)

    print()
    print(f"Issues examined: {args.max_issues}")
    print(f"Comments collected: {len(comments)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()