from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


GITHUB_API_VERSION = "2022-11-28"


def parse_next_link(link_header: str | None) -> str | None:
    """Extract the next-page URL from a GitHub API Link header."""

    if not link_header:
        return None

    for section in link_header.split(","):
        parts = [part.strip() for part in section.split(";")]

        if len(parts) < 2:
            continue

        relation_parts = parts[1:]

        if 'rel="next"' in relation_parts:
            return parts[0].strip("<>")

    return None


def create_headers(token: str | None = None) -> dict[str, str]:
    """Create headers for GitHub API requests."""

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": "NeedSignal-Research-Collector",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def transform_issue(
    issue: dict[str, Any],
    owner: str,
    repo: str,
) -> dict[str, Any]:
    """Convert one GitHub issue into a clean dataset row."""

    labels = [
        label.get("name", "")
        for label in issue.get("labels", [])
        if isinstance(label, dict)
    ]

    body = issue.get("body") or ""

    return {
        "repository": f"{owner}/{repo}",
        "issue_number": issue.get("number"),
        "title": issue.get("title", ""),
        "body": body,
        "state": issue.get("state"),
        "state_reason": issue.get("state_reason"),
        "labels": " | ".join(labels),
        "comments_count": issue.get("comments", 0),
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
        "body_length": len(body),
        "issue_url": issue.get("html_url"),
    }


def collect_issues(
    owner: str,
    repo: str,
    limit: int = 100,
) -> pd.DataFrame:
    """Collect real issues from a public GitHub repository."""

    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    load_dotenv()

    token = os.getenv("GITHUB_TOKEN")
    headers = create_headers(token)

    url: str | None = (
        f"https://api.github.com/repos/{owner}/{repo}/issues"
    )

    params: dict[str, Any] | None = {
        "state": "all",
        "sort": "created",
        "direction": "desc",
        "per_page": 100,
    }

    rows: list[dict[str, Any]] = []

    while url and len(rows) < limit:
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
                f"GitHub request failed with status "
                f"{response.status_code}. "
                f"Remaining requests: {remaining}. "
                f"Response: {response.text[:300]}"
            ) from error

        results = response.json()

        if not isinstance(results, list):
            raise RuntimeError(
                "GitHub returned an unexpected response format."
            )

        for issue in results:
            # GitHub's issues endpoint also returns pull requests.
            if "pull_request" in issue:
                continue

            rows.append(transform_issue(issue, owner, repo))

            if len(rows) >= limit:
                break

        url = parse_next_link(response.headers.get("Link"))

        # The next-page URL already includes its own query parameters.
        params = None

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect GitHub issues for NeedSignal."
    )

    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output")

    args = parser.parse_args()

    dataframe = collect_issues(
        owner=args.owner,
        repo=args.repo,
        limit=args.limit,
    )

    output_path = (
        Path(args.output)
        if args.output
        else Path("data/raw")
        / f"{args.owner}_{args.repo}_issues.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    print()
    print(f"Repository: {args.owner}/{args.repo}")
    print(f"Issues collected: {len(dataframe)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()