"""Pytest suite for src.goodreads_etl.utils.book_recorder."""

from __future__ import annotations

import polars as pl
import pytest

from src.goodreads_etl.utils.book_recorder import (
    BOOK_RECORD_NUMERIC_COLUMNS,
    BOOK_RECORD_SCHEMA,
    BookRecord,
)


REQUIRED_FIELDS = {
    "book_id": "1",
    "url": "https://www.goodreads.com/book/show/1",
    "title": "Some Title",
    "first_author": "Some Author",
    "first_author_url": "https://www.goodreads.com/author/show/1",
}


class TestBookRecordDefaults:
    """Tests for initialization and default values of `BookRecord` dataclass fields."""

    def test_required_fields_are_set(self) -> None:
        """Verify that mandatory fields are correctly assigned upon initialization."""
        record = BookRecord(**REQUIRED_FIELDS)
        assert record.book_id == "1"
        assert record.title == "Some Title"

    def test_numeric_fields_default_correctly(self) -> None:
        """Verify that unprovided numeric attributes default to zero or zero float."""
        record = BookRecord(**REQUIRED_FIELDS)
        assert record.first_author_num_books == 0
        assert record.first_author_num_followers == 0
        assert record.average_rating == 0.0
        assert record.num_reviews == 0
        assert record.num_pages == 0
        assert record.num_currently_reading == 0
        assert record.num_want_to_read == 0

    def test_optional_fields_default_to_none(self) -> None:
        """Verify that optional string and metadata fields default to None."""
        record = BookRecord(**REQUIRED_FIELDS)
        assert record.first_published is None
        assert record.publisher is None
        assert record.language_code is None
        assert record.description is None
        assert record.genres is None
        assert record.format is None
        assert record.series is None

    def test_missing_required_field_raises_type_error(self) -> None:
        """Verify TypeError is raised when a required parameter is omitted during instantiation."""
        with pytest.raises(TypeError):
            BookRecord(book_id="1", url="u", title="t", first_author="a")


class TestBookRecordToDict:
    """Tests for the `BookRecord.to_dict()` conversion method."""

    def test_to_dict_returns_all_fields(self) -> None:
        """Verify `to_dict()` extracts every dataclass attribute and keys match the schema."""
        record = BookRecord(**REQUIRED_FIELDS)
        result = record.to_dict()
        assert result["book_id"] == "1"
        assert result["num_pages"] == 0
        assert result["description"] is None
        assert set(result.keys()) == set(BOOK_RECORD_SCHEMA.keys())

    def test_to_dict_reflects_custom_values(self) -> None:
        """Verify `to_dict()` correctly represents overridden non-default values."""
        record = BookRecord(**REQUIRED_FIELDS, num_pages=320, genres="Fiction, Drama")
        result = record.to_dict()
        assert result["num_pages"] == 320
        assert result["genres"] == "Fiction, Drama"


class TestBookRecordSlotsAndImmutability:
    """Tests evaluating memory slots configuration and attribute mutability."""

    def test_uses_slots_rejects_new_attributes(self) -> None:
        """Verify `slots=True` prevents dynamically setting undeclared attributes."""
        record = BookRecord(**REQUIRED_FIELDS)
        with pytest.raises(AttributeError):
            record.some_new_attr = "value"

    def test_is_mutable_not_frozen(self) -> None:
        """Verify that existing fields remain mutable when `frozen=False`."""
        # slots=True without frozen=True -> normal attribute assignment works.
        record = BookRecord(**REQUIRED_FIELDS)
        record.num_pages = 500
        assert record.num_pages == 500


class TestBookRecordSchema:
    """Tests validating `BOOK_RECORD_SCHEMA` and integration with Polars DataFrames."""

    def test_schema_keys_match_bookrecord_fields(self) -> None:
        """Verify schema dictionary keys align perfectly with `BookRecord` attributes."""
        record = BookRecord(**REQUIRED_FIELDS)
        assert set(BOOK_RECORD_SCHEMA.keys()) == set(record.to_dict().keys())

    def test_schema_values_are_polars_dtypes(self) -> None:
        """Verify every schema value maps to a valid Polars data type (Utf8, Int64, or Float64)."""
        for column, dtype in BOOK_RECORD_SCHEMA.items():
            assert dtype in (pl.Utf8, pl.Int64, pl.Float64), column

    def test_numeric_columns_are_subset_of_schema(self) -> None:
        """Verify `BOOK_RECORD_NUMERIC_COLUMNS` contains only keys present in the primary schema."""
        assert set(BOOK_RECORD_NUMERIC_COLUMNS).issubset(BOOK_RECORD_SCHEMA.keys())

    def test_numeric_columns_have_int_or_float_dtype(self) -> None:
        """Verify all tracked numeric column names map to Int64 or Float64 dtypes."""
        for column in BOOK_RECORD_NUMERIC_COLUMNS:
            assert BOOK_RECORD_SCHEMA[column] in (pl.Int64, pl.Float64)

    def test_schema_can_build_empty_polars_dataframe(self) -> None:
        """Verify Polars DataFrame instantiates cleanly using `BOOK_RECORD_SCHEMA` as schema."""
        df = pl.DataFrame(schema=BOOK_RECORD_SCHEMA)
        assert df.columns == list(BOOK_RECORD_SCHEMA.keys())
        assert df.height == 0

    def test_schema_can_construct_dataframe_from_record(self) -> None:
        """Verify Polars DataFrame correctly parses a list containing a converted `BookRecord` dict."""
        record = BookRecord(**REQUIRED_FIELDS, num_pages=100)
        df = pl.DataFrame([record.to_dict()], schema=BOOK_RECORD_SCHEMA)
        assert df.height == 1
        assert df["num_pages"][0] == 100