"""TRANSFORM stage: BookRecords -> a schema-aligned Polars DataFrame."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import polars as pl

from src.goodreads_etl.utils.book_recorder import (
    BOOK_RECORD_NUMERIC_COLUMNS,
    BOOK_RECORD_SCHEMA,
    BookRecord,
)


def _records_to_column_dict(
    *,
    records: list[BookRecord],
    columns: list[str],
) -> dict[str, list[object]]:
    """Pivot a list of BookRecords into a column-major dict for Polars.

    Converts row-based `BookRecord` objects into a dictionary mapping each
    column name to a list of its corresponding values across all records.

    Args:
        records: A list of `BookRecord` instances to convert.
        columns: The ordered list of target schema column names.

    Returns:
        A dictionary where keys are column names and values are lists of column entries.
    """
    rows = [record.to_dict() for record in records]
    return {column: [row.get(column) for row in rows] for column in columns}


def _fill_missing_numeric_values(
    *,
    dataframe: pl.DataFrame,
    numeric_columns: tuple[str, ...],
) -> pl.DataFrame:
    """Replace nulls in numeric columns with 0.

    In this dataset, a missing count represents a zero count rather than an
    unknown value (e.g., missing review count implies zero reviews).

    Args:
        dataframe: The Polars DataFrame to transform.
        numeric_columns: A tuple of column names where null values should be zero-filled.

    Returns:
        A new Polars DataFrame with missing numeric values replaced by 0.
    """
    return dataframe.with_columns(
        [pl.col(column).fill_null(0).alias(column) for column in numeric_columns]
    )


def records_to_dataframe(
    *,
    records: list[BookRecord],
    schema: Mapping[str, type[pl.DataType]] = BOOK_RECORD_SCHEMA,
    numeric_columns: tuple[str, ...] = BOOK_RECORD_NUMERIC_COLUMNS,
) -> pl.DataFrame:
    """Build a schema-enforced Polars DataFrame from a batch of BookRecords.

    Pivots row-based book records into a column-oriented structure, casts
    columns to the defined Polars data types, fills missing numeric counts
    with zeros, and enforces strict column ordering.

    This is the single public entrypoint for the transform stage.

    Args:
        records: A list of `BookRecord` instances to transform.
        schema: A mapping of column names to Polars `DataType` classes.
            Defaults to `BOOK_RECORD_SCHEMA`.
        numeric_columns: A tuple of numeric column names to fill nulls with 0.
            Defaults to `BOOK_RECORD_NUMERIC_COLUMNS`.

    Returns:
        A schema-aligned Polars DataFrame containing the transformed record data.
    """
    columns = list(schema.keys())

    if not records:
        return pl.DataFrame(schema=schema)

    column_data = _records_to_column_dict(records=records, columns=columns)
    dataframe = pl.DataFrame(column_data).cast(cast(pl.Schema, dict(schema)))
    dataframe = _fill_missing_numeric_values(dataframe=dataframe, numeric_columns=numeric_columns)

    return dataframe.select(columns)
