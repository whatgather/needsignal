from needsignal.collect import parse_next_link


def test_parse_next_link_returns_next_url():
    header = (
        '<https://api.github.com/repos/example/issues?page=2>; rel="next", '
        '<https://api.github.com/repos/example/issues?page=5>; rel="last"'
    )

    result = parse_next_link(header)

    assert result == "https://api.github.com/repos/example/issues?page=2"


def test_parse_next_link_returns_none_without_next():
    header = (
        '<https://api.github.com/repos/example/issues?page=1>; rel="first", '
        '<https://api.github.com/repos/example/issues?page=5>; rel="last"'
    )

    assert parse_next_link(header) is None


def test_parse_next_link_returns_none_for_empty_header():
    assert parse_next_link(None) is None
    assert parse_next_link("") is None