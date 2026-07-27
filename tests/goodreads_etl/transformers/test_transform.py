"""Pytest suite for src.goodreads_etl.transformers.transform."""

from __future__ import annotations

from typing import Any

import polars as pl

from src.goodreads_etl.transformers.transform import (
    _fill_missing_numeric_values,
    _records_to_column_dict,
    records_to_dataframe,
)
from src.goodreads_etl.utils.book_recorder import BOOK_RECORD_SCHEMA, BookRecord


def make_record(**overrides: Any) -> BookRecord:
    """Factory helper to construct a dummy `BookRecord` instance.

    Args:
        **overrides: Optional field overrides for default book record attributes.

    Returns:
        BookRecord: A sample BookRecord instance populated with test data.
    """
    defaults = dict(
        book_id="1",
        url="https://www.goodreads.com/book/show/1",
        title="Dune",
        first_author="Frank Herbert",
        first_author_url="https://www.goodreads.com/author/show/1",
    )
    defaults.update(overrides)
    return BookRecord(**defaults)


# ---------------------------------------------------------------------------
# _records_to_column_dict
# ---------------------------------------------------------------------------


class TestRecordsToColumnDict:
    """Tests evaluating dictionary transformation from record lists."""

    def test_restricts_output_to_requested_columns_only(self) -> None:
        """Verify output dictionary is strictly limited to requested column keys."""
        records = [make_record(book_id="1", title="Dune")]
        result = _records_to_column_dict(records=records, columns=["book_id", "not_a_real_field"])
        assert list(result.keys()) == ["book_id", "not_a_real_field"]
        assert result == {"book_id": ["1"], "not_a_real_field": [None]}


# ---------------------------------------------------------------------------
# _fill_missing_numeric_values
# ---------------------------------------------------------------------------


class TestFillMissingNumericValues:
    """Tests evaluating Polars DataFrame numeric null imputation logic."""

    def test_replaces_nulls_with_zero_in_specified_columns(self) -> None:
        """Verify null values are imputed to zero only within designated target numeric columns."""
        dataframe = pl.DataFrame(
            {
                "num_reviews": [None, 5, None],
                "title": ["a", "b", None],
                "num_pages": [10, None, 200],
            }
        )
        result = _fill_missing_numeric_values(
            dataframe=dataframe, numeric_columns=("num_reviews", "num_pages")
        )
        assert result["num_reviews"].to_list() == [0, 5, 0]
        assert result["title"].to_list() == ["a", "b", None]
        assert result["num_pages"].to_list() == [10, 0, 200]


# ---------------------------------------------------------------------------
# records_to_dataframe (public entrypoint)
# ---------------------------------------------------------------------------


class TestRecordsToDataframe:
    """Tests evaluating top-level DataFrame construction and normalization pipeline."""

    def test_preserves_and_fills_fields(self) -> None:
        """Verify records are converted to Polars DataFrame with full schema compliance and null filling."""
        record1 = make_record(
            book_id="13",
            title="Dune",
            num_reviews=None,
            num_pages=None,
            publisher=None,
        )
        record2 = make_record(book_id="42", title="Hyperion", average_rating=4.2)
        dataframe = records_to_dataframe(records=[record1, record2])

        assert dataframe.columns == list(BOOK_RECORD_SCHEMA.keys())
        assert dataframe.schema == BOOK_RECORD_SCHEMA
        assert dataframe.height == 2

        assert dataframe["book_id"].to_list() == ["13", "42"]
        assert dataframe["title"].to_list() == ["Dune", "Hyperion"]
        assert dataframe["num_reviews"].to_list() == [0, 0]
        assert dataframe["num_pages"].to_list() == [0, 0]
        assert dataframe["average_rating"].to_list() == [0, 4.2]
        assert dataframe["publisher"].to_list() == [None, None]

    def test_respects_custom_schema_and_numeric_columns_overrides(self) -> None:
        """Verify custom schema mappings and target numeric column overrides are honored."""
        custom_schema = {"book_id": pl.Utf8, "num_reviews": pl.Int64}
        custom_numeric_columns = ("num_reviews",)
        record = make_record(book_id="1", num_reviews=None)

        dataframe = records_to_dataframe(
            records=[record],
            schema=custom_schema,
            numeric_columns=custom_numeric_columns,
        )

        assert dataframe.columns == ["book_id", "num_reviews"]
        assert dataframe["num_reviews"].to_list() == [0]

    def test_empty_records_respects_custom_schema_override(self) -> None:
        """Verify empty input record list produces zero-row DataFrame with specified custom schema."""
        custom_schema = {"book_id": pl.Utf8}
        dataframe = records_to_dataframe(records=[], schema=custom_schema)

        assert dataframe.columns == ["book_id"]
        assert dataframe.height == 0