"""Pure helpers for normalizing text and parsing simple numeric values.

No dependency on any other project module -- this is the lowest layer.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Sequence

DEFAULT_PARENTHETICAL_PATTERN = re.compile(r"\s*\([^()]*\)")
DEFAULT_WHITESPACE_PATTERN = re.compile(r"\s+")
DEFAULT_INTEGER_PATTERN = re.compile(r"\d[\d,]*")
DEFAULT_FLOAT_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def clean_object(
    *,
    value: object | None,
    parenthetical_pattern: re.Pattern[str] = DEFAULT_PARENTHETICAL_PATTERN,
    whitespace_pattern: re.Pattern[str] = DEFAULT_WHITESPACE_PATTERN,
) -> str | None:
    """Return normalized text, or ``None`` if no meaningful text remains.

    Unescapes HTML entities, applies Unicode NFKC normalization, strips
    byte-order marks and control characters, removes parenthetical asides
    (e.g. "(2020 edition)"), lowercases, and collapses whitespace.

    Args:
        value: The raw string or object to normalize. If None, returns None.
        parenthetical_pattern: Compiled regex pattern used to strip parenthetical text.
            Defaults to matching balanced non-nested parentheses.
        whitespace_pattern: Compiled regex pattern used to collapse multiple whitespace characters.
            Defaults to matching one or more whitespace characters.

    Returns:
        The normalized text string, or None if the input is None or becomes empty after cleaning.
    """
    if value is None:
        return None

    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufeff", "")
    text = text.lower()
    text = parenthetical_pattern.sub("", text)
    text = "".join(
        character
        for character in text
        if unicodedata.category(character)[0] != "C" or character in "\n\t"
    )
    text = whitespace_pattern.sub(" ", text).strip()

    return text or None


def get_first_match(
    *,
    text: str | None,
    patterns: Sequence[re.Pattern[str]],
    group: int = 1,
) -> str | None:
    """Return a capturing group from the first pattern that matches.

    Patterns are tried in order; the first one that matches wins, which
    lets callers pass a most-specific-first, most-general-last pattern list.

    Args:
        text: The target string to search within. If None or empty, returns None.
        patterns: A sequence of compiled regex patterns to test against the text in order.
        group: The capturing group index to extract from the successful match. Defaults to 1.

    Returns:
        The captured string group from the first matching pattern, or None if no patterns match
        or if the text is empty/None.
    """
    if not text:
        return None

    for pattern in patterns:
        match = pattern.search(text)
        if match is not None:
            return match.group(group)

    return None


def get_int(
    *,
    text: str | None,
    integer_pattern: re.Pattern[str] = DEFAULT_INTEGER_PATTERN,
) -> int | None:
    """Extract the first integer in ``text``, thousands separators OK.

    Args:
        text: The target string containing the target integer. If None, returns None.
        integer_pattern: Compiled regex pattern used to locate the integer sequence.
            Defaults to matching digits with optional commas.

    Returns:
        The extracted integer, or None if text is None or no integer match is found.
    """
    if text is None:
        return None

    match = integer_pattern.search(text)
    if match is None:
        return None

    return int(match.group(0).replace(",", ""))


def get_float(
    *,
    text: str | None,
    float_pattern: re.Pattern[str] = DEFAULT_FLOAT_PATTERN,
) -> float | None:
    """Extract the first decimal number in ``text``.

    Args:
        text: The target string containing the target decimal value. If None, returns None.
        float_pattern: Compiled regex pattern used to locate the float sequence.
            Defaults to matching whole and optional fractional digits.

    Returns:
        The extracted float value, or None if text is None or no float match is found.
    """
    if text is None:
        return None

    match = float_pattern.search(text)
    if match is None:
        return None

    return float(match.group(0))
