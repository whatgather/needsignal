
def test_parse_next_link_returns_next_url() -> None:
    header = (
        '<https://api.github.com/repositories/1/issues?page=2>; rel="next", '
        '<https://api.github.com/repositories/1/issues?page=4>; rel="last"'
    )
    assert (
        parse_next_link(header)
        == "https://api.github.com/repositories/1/issues?page=2"
    )


def test_parse_next_link_returns_none_without_next() -> None:
    header = '<https://api.github.com/repositories/1/issues?page=1>; rel="first"'
    assert parse_next_link(header) is None
