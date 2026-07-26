from typing import Optional


def parse_next_link(link_header: Optional[str]) -> Optional[str]:
    """Extract the next-page URL from a GitHub API Link header."""

    if not link_header:
        return None

    for section in link_header.split(","):
        parts = section.strip().split(";")

        if len(parts) < 2:
            continue

        url_part = parts[0].strip()
        relation = parts[1].strip()

        if relation == 'rel="next"':
            return url_part.strip("<>")

    return None
{
  "repositories": [
    "n8n-io/n8n",
    "activepieces/activepieces",
    "node-red/node-red"
  ]
}
