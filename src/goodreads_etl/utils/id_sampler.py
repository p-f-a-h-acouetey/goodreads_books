"""Random Goodreads book-ID sampling, with exclusion-aware retry."""

from __future__ import annotations

from random import randint


def _count_excluded_ids_in_range(
    *,
    excluded_book_ids: set[str],
    min_book_id: int,
    max_book_id: int,
) -> int:
    """Return how many excluded IDs fall inside [min_book_id, max_book_id].

    Args:
        excluded_book_ids: Set of string IDs that should be ignored or omitted.
        min_book_id: The lower bound (inclusive) of the target ID range.
        max_book_id: The upper bound (inclusive) of the target ID range.

    Returns:
        The total count of valid numeric string IDs from ``excluded_book_ids``
        that fall within the specified inclusive range.
    """
    return sum(
        1
        for book_id_str in excluded_book_ids
        if book_id_str.isdigit() and min_book_id <= int(book_id_str) <= max_book_id
    )


def _draw_unused_id(
    *,
    min_book_id: int,
    max_book_id: int,
    excluded_book_ids: set[str],
    max_attempts: int,
) -> str:
    """Randomly draw candidate IDs until one is not in ``excluded_book_ids``.

    Args:
        min_book_id: The lower bound (inclusive) for the random integer draw.
        max_book_id: The upper bound (inclusive) for the random integer draw.
        excluded_book_ids: Set of string IDs already used or blacklisted.
        max_attempts: Maximum number of random draws allowed before failing.

    Returns:
        A unique string ID guaranteed not to be in ``excluded_book_ids``.

    Raises:
        RuntimeError: If no unused ID is successfully drawn within ``max_attempts``.
    """
    for _ in range(max_attempts):
        candidate_id = str(randint(min_book_id, max_book_id))
        if candidate_id not in excluded_book_ids:
            return candidate_id

    raise RuntimeError(
        f"Could not draw an unused book ID after {max_attempts} attempts. "
        "Increase max_attempts or widen the ID range."
    )


def generate_random_book_id(
    *,
    min_book_id: int,
    max_book_id: int,
    excluded_book_ids: set[str] | None = None,
    max_attempts: int = 10_000,
) -> str:
    """Return an unused random book ID from an inclusive range.

    Side-effect free: does not modify ``excluded_book_ids``. The caller
    must record the returned ID immediately to avoid duplicate draws
    across concurrent crawl workers.

    Args:
        min_book_id: The lower bound (inclusive) of the allowable ID range.
        max_book_id: The upper bound (inclusive) of the allowable ID range.
        excluded_book_ids: Optional set of string IDs to skip. Defaults to None.
        max_attempts: Maximum number of random draws before timing out. Defaults to 10,000.

    Returns:
        A string representation of a randomly selected book ID that does not exist
        within ``excluded_book_ids``.

    Raises:
        RuntimeError: If all IDs in the specified range are already excluded,
            or if a free ID cannot be drawn within ``max_attempts``.
    """
    excluded_ids = excluded_book_ids or set()
    range_size = max_book_id - min_book_id + 1

    excluded_in_range = _count_excluded_ids_in_range(
        excluded_book_ids=excluded_ids,
        min_book_id=min_book_id,
        max_book_id=max_book_id,
    )
    if excluded_in_range >= range_size:
        raise RuntimeError(
            f"All IDs in range {min_book_id}-{max_book_id} are excluded. "
            "Use a wider range or clear the persisted ID tracker file."
        )

    return _draw_unused_id(
        min_book_id=min_book_id,
        max_book_id=max_book_id,
        excluded_book_ids=excluded_ids,
        max_attempts=max_attempts,
    )