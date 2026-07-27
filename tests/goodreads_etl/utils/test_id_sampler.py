"""Pytest suite for src.goodreads_etl.utils.id_sampler."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.goodreads_etl.utils.id_sampler import (
    _count_excluded_ids_in_range,
    _draw_unused_id,
    generate_random_book_id,
)


# ---------------------------------------------------------------------------
# _count_excluded_ids_in_range
# ---------------------------------------------------------------------------

class TestCountExcludedIdsInRange:
    """Tests for the `_count_excluded_ids_in_range` helper function."""

    def test_returns_zero_for_empty_set(self) -> None:
        """Verify that an empty exclusion set returns a count of zero."""
        assert _count_excluded_ids_in_range(
            excluded_book_ids=set(), min_book_id=1, max_book_id=100
        ) == 0

    def test_counts_only_in_range_numeric_ids(self) -> None:
        """Verify that non-digit strings, invalid numbers, and out-of-range IDs are ignored."""
        excluded = {
            "1", "5", "10", "50", "100",  # inside range (incl. boundaries)
            "abc", "12a", "", "1.5", "-5",  # not valid digit strings
            "500", "999",  # outside range
        }
        assert _count_excluded_ids_in_range(
            excluded_book_ids=excluded, min_book_id=1, max_book_id=100
        ) == 5


# ---------------------------------------------------------------------------
# _draw_unused_id
# ---------------------------------------------------------------------------

class TestDrawUnusedId:
    """Tests for the private `_draw_unused_id` rejection-sampling loop."""

    def test_returns_first_unused_draw(self) -> None:
        """Verify immediate return on first attempt when the sampled ID is not excluded."""
        with patch("src.goodreads_etl.utils.id_sampler.randint", return_value=3) as mock_randint:
            result = _draw_unused_id(
                min_book_id=1, max_book_id=10, excluded_book_ids=set(), max_attempts=100
            )
        assert result == "3"
        mock_randint.assert_called_once_with(1, 10)

    def test_retries_past_excluded_ids_until_unused_one_found(self) -> None:
        """Verify rejection sampling continues until a non-excluded ID is generated."""
        with patch("src.goodreads_etl.utils.id_sampler.randint", side_effect=[5, 5, 7]):
            result = _draw_unused_id(
                min_book_id=1, max_book_id=10, excluded_book_ids={"5"}, max_attempts=10
            )
        assert result == "7"

    def test_raises_runtime_error_with_attempt_count_when_exhausted(self) -> None:
        """Verify RuntimeError is raised when `max_attempts` threshold is exceeded."""
        with patch("src.goodreads_etl.utils.id_sampler.randint", return_value=5) as mock_randint:
            with pytest.raises(RuntimeError, match="after 3 attempts"):
                _draw_unused_id(
                    min_book_id=1, max_book_id=10, excluded_book_ids={"5"}, max_attempts=3
                )
        assert mock_randint.call_count == 3


# ---------------------------------------------------------------------------
# generate_random_book_id
# ---------------------------------------------------------------------------

class TestGenerateRandomBookId:
    """Tests for the public `generate_random_book_id` entrypoint."""

    def test_returns_valid_id_within_range_no_exclusions(self) -> None:
        """Verify generated ID is a numeric string falling within specified bounds."""
        result = generate_random_book_id(min_book_id=1, max_book_id=1000)
        assert result.isdigit()
        assert 1 <= int(result) <= 1000

    def test_none_and_empty_excluded_set_behave_identically(self) -> None:
        """Verify passing None vs empty set for `excluded_book_ids` yields identical results."""
        result_with_none = generate_random_book_id(min_book_id=5, max_book_id=5, excluded_book_ids=None)
        result_with_empty = generate_random_book_id(min_book_id=5, max_book_id=5, excluded_book_ids=set())
        assert result_with_none == result_with_empty == "5"

    def test_excludes_provided_ids(self) -> None:
        """Verify excluded IDs are skipped when drawing a random ID."""
        with patch("src.goodreads_etl.utils.id_sampler.randint", side_effect=[1, 1, 2]):
            result = generate_random_book_id(
                min_book_id=1, max_book_id=2, excluded_book_ids={"1"}, max_attempts=10
            )
        assert result == "2"

    def test_exclusions_outside_range_are_ignored(self) -> None:
        """Verify exclusion sets containing IDs outside min/max range do not affect generation."""
        result = generate_random_book_id(
            min_book_id=1, max_book_id=1, excluded_book_ids={"999", "abc"}
        )
        assert result == "1"

    def test_raises_when_every_id_in_range_is_excluded(self) -> None:
        """Verify RuntimeError is immediately raised if all candidate IDs in range are excluded."""
        # Boundary case: excluded_in_range == range_size triggers the ">="
        # exhaustion check exactly, not "> range_size".
        excluded = {"1", "2", "3", "4"}
        with pytest.raises(RuntimeError, match="All IDs in range 1-4 are excluded"):
            generate_random_book_id(min_book_id=1, max_book_id=4, excluded_book_ids=excluded)

    def test_raises_when_max_attempts_exhausted_despite_range_having_room(self) -> None:
        """Verify RuntimeError when sampling fails to hit an available ID within max attempts."""
        with patch("src.goodreads_etl.utils.id_sampler.randint", return_value=1):
            with pytest.raises(RuntimeError, match="Could not draw an unused book ID"):
                generate_random_book_id(
                    min_book_id=1, max_book_id=3, excluded_book_ids={"1"}, max_attempts=5
                )

    def test_delegates_to_draw_unused_id_with_correct_args(self) -> None:
        """Verify `generate_random_book_id` delegates parameter handling to `_draw_unused_id`."""
        with patch("src.goodreads_etl.utils.id_sampler._draw_unused_id", return_value="7") as mock_draw:
            result = generate_random_book_id(
                min_book_id=1, max_book_id=10, excluded_book_ids={"3"}, max_attempts=25
            )
        assert result == "7"
        mock_draw.assert_called_once_with(
            min_book_id=1, max_book_id=10, excluded_book_ids={"3"}, max_attempts=25
        )