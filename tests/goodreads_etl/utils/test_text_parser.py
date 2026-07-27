"""Pytest suite for src.goodreads_etl.utils.text_parser."""

from __future__ import annotations

import re

from src.goodreads_etl.utils.text_parser import (
    clean_object,
    get_first_match,
    get_float,
    get_int,
)


class TestCleanObject:
    """Tests for `clean_object` text normalization and sanitization function."""

    def test_returns_none_for_none_input(self) -> None:
        """Verify None is returned when input value is None."""
        assert clean_object(value=None) is None

    def test_lowercases_text(self) -> None:
        """Verify casing is converted to lowercase."""
        assert clean_object(value="Frank Herbert") == "frank herbert"

    def test_unescapes_html_entities(self) -> None:
        """Verify HTML entities (e.g. `&amp;`) are decoded to plain text."""
        assert clean_object(value="Frank &amp; Herbert") == "frank & herbert"

    def test_normalizes_unicode_nfkc(self) -> None:
        """Verify string is normalized using Unicode NFKC form."""
        assert clean_object(value="Caf\u00e9") == "caf\u00e9"

    def test_replaces_nbsp_with_regular_space(self) -> None:
        """Verify non-breaking space characters (`\\u00a0`) are replaced by standard spaces."""
        assert clean_object(value="Frank\u00a0Herbert") == "frank herbert"

    def test_strips_byte_order_mark(self) -> None:
        """Verify Byte Order Mark (`\\ufeff`) prefixes are stripped."""
        assert clean_object(value="\ufeffFrank Herbert") == "frank herbert"

    def test_removes_parenthetical_asides(self) -> None:
        """Verify parenthetical commentary in parentheses is removed."""
        assert clean_object(value="Dune (2020 edition)") == "dune"

    def test_removes_multiple_parentheticals(self) -> None:
        """Verify multiple parenthetical expressions are stripped entirely."""
        assert clean_object(value="Dune (2020) (reprint)") == "dune"

    def test_strips_control_characters(self) -> None:
        """Verify non-printable control characters (e.g., `\\x00`) are removed."""
        assert clean_object(value="Frank\x00Herbert") == "frankherbert"

    def test_preserves_newline_and_tab_control_chars(self) -> None:
        """Verify newline (`\\n`) and tab (`\\t`) characters are converted to spaces."""
        result = clean_object(value="Frank\nHerbert\tAuthor")
        assert result == "frank herbert author"

    def test_collapses_multiple_whitespace(self) -> None:
        """Verify redundant whitespace sequences are collapsed into single spaces."""
        assert clean_object(value="Frank    Herbert ") == "frank herbert"

    def test_returns_none_when_result_is_empty_string(self) -> None:
        """Verify None is returned if cleaned string consists solely of whitespace."""
        assert clean_object(value="   ") is None

    def test_returns_none_when_only_parenthetical_content(self) -> None:
        """Verify None is returned when string content is completely inside parentheses."""
        assert clean_object(value="(2020 edition)") is None

    def test_converts_non_string_object_to_string(self) -> None:
        """Verify non-string objects (e.g., integers) are cast to string before cleaning."""
        assert clean_object(value=412) == "412"

    def test_custom_parenthetical_pattern(self) -> None:
        """Verify custom regex pattern overrides default parenthetical stripping."""
        custom_pattern = re.compile(r"\s*\[[^\[\]]*\]")
        result = clean_object(value="Dune [2020 edition]", parenthetical_pattern=custom_pattern)
        assert result == "dune"

    def test_custom_whitespace_pattern(self) -> None:
        """Verify custom regex pattern overrides default whitespace normalization."""
        custom_pattern = re.compile(r"-+")
        result = clean_object(value="frank-herbert", whitespace_pattern=custom_pattern)
        assert result == "frank herbert"


class TestGetFirstMatch:
    """Tests for `get_first_match` pattern matching utility."""

    def test_returns_none_for_none_text(self) -> None:
        """Verify None is returned when target input text is None."""
        patterns = (re.compile(r"(\d+)"),)
        assert get_first_match(text=None, patterns=patterns) is None

    def test_returns_none_for_empty_text(self) -> None:
        """Verify None is returned when target input text is an empty string."""
        patterns = (re.compile(r"(\d+)"),)
        assert get_first_match(text="", patterns=patterns) is None

    def test_returns_match_from_first_matching_pattern(self) -> None:
        """Verify matched capture group string is returned on first pattern match."""
        patterns = (re.compile(r"(\d+)\s+pages"),)
        assert get_first_match(text="350 pages", patterns=patterns) == "350"

    def test_tries_patterns_in_order_first_wins(self) -> None:
        """Verify candidate regex patterns are evaluated in order and first match wins."""
        patterns = (
            re.compile(r"first published (\w+)"),
            re.compile(r"published (\w+)"),
        )
        assert get_first_match(text="first published august", patterns=patterns) == "august"

    def test_falls_back_to_second_pattern_when_first_fails(self) -> None:
        """Verify fallback to subsequent regex pattern when preceding patterns yield no match."""
        patterns = (
            re.compile(r"first published (\w+)"),
            re.compile(r"published (\w+)"),
        )
        assert get_first_match(text="published august", patterns=patterns) == "august"

    def test_returns_none_when_no_pattern_matches(self) -> None:
        """Verify None is returned when input text matches none of the patterns."""
        patterns = (re.compile(r"(\d+)\s+pages"),)
        assert get_first_match(text="nothing relevant", patterns=patterns) is None

    def test_returns_none_when_patterns_sequence_empty(self) -> None:
        """Verify None is returned when passed an empty tuple/sequence of patterns."""
        assert get_first_match(text="some text", patterns=()) is None

    def test_uses_custom_group_index(self) -> None:
        """Verify explicit capture group index selects designated sub-match."""
        patterns = (re.compile(r"(\w+)\s+(\d+)"),)
        assert get_first_match(text="pages 350", patterns=patterns, group=2) == "350"


class TestGetInt:
    """Tests for `get_int` integer extraction utility."""

    def test_returns_none_for_none_text(self) -> None:
        """Verify None is returned when input text is None."""
        assert get_int(text=None) is None

    def test_extracts_simple_integer(self) -> None:
        """Verify simple integer digit strings are extracted and cast to `int`."""
        assert get_int(text="350 pages") == 350

    def test_extracts_integer_with_thousands_separator(self) -> None:
        """Verify integers with comma thousands separators are parsed correctly."""
        assert get_int(text="1,234,789 reviews") == 1234789

    def test_returns_none_when_no_integer_present(self) -> None:
        """Verify None is returned when text contains no numeric digit sequences."""
        assert get_int(text="no numbers here") is None

    def test_extracts_first_integer_when_multiple_present(self) -> None:
        """Verify only the first integer encountered in input text is extracted."""
        assert get_int(text="42 books, 10,000 followers") == 42

    def test_custom_integer_pattern(self) -> None:
        """Verify custom regex pattern overrides default integer regex logic."""
        custom_pattern = re.compile(r"\d+")
        assert get_int(text="Book #7 of series", integer_pattern=custom_pattern) == 7


class TestGetFloat:
    """Tests for `get_float` floating-point number extraction utility."""

    def test_returns_none_for_none_text(self) -> None:
        """Verify None is returned when input text is None."""
        assert get_float(text=None) is None

    def test_extracts_whole_number_as_float(self) -> None:
        """Verify whole numbers and decimal values are both correctly parsed into `float`."""
        assert get_float(text="rated 4 stars") == 4.0
        assert get_float(text="rated 4.5 stars") == 4.5

    def test_returns_none_when_no_float_present(self) -> None:
        """Verify None is returned when text contains no floating-point numbers."""
        assert get_float(text="no numbers here") is None

    def test_extracts_first_float_when_multiple_present(self) -> None:
        """Verify only the first floating-point number in text is extracted."""
        assert get_float(text="4.5 out of 5.0") == 4.5

    def test_custom_float_pattern(self) -> None:
        """Verify custom regex pattern overrides default floating-point regex logic."""
        custom_pattern = re.compile(r"\d+\.\d+")
        assert get_float(text="Price: $19.99", float_pattern=custom_pattern) == 19.99
