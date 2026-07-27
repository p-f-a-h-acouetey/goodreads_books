"""Book record schema: the normalized data contract for the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

BOOK_RECORD_SCHEMA: dict[str, type[pl.DataType]] = {
    "book_id": pl.Utf8,
    "url": pl.Utf8,
    "title": pl.Utf8,
    "first_author": pl.Utf8,
    "first_author_url": pl.Utf8,
    "first_author_num_books": pl.Int64,
    "first_author_num_followers": pl.Int64,
    "average_rating": pl.Float64,
    "num_reviews": pl.Int64,
    "first_published": pl.Utf8,
    "publisher": pl.Utf8,
    "language_code": pl.Utf8,
    "num_pages": pl.Int64,
    "description": pl.Utf8,
    "genres": pl.Utf8,
    "format": pl.Utf8,
    "series": pl.Utf8,
    "num_currently_reading": pl.Int64,
    "num_want_to_read": pl.Int64,
}

BOOK_RECORD_NUMERIC_COLUMNS: tuple[str, ...] = (
    "first_author_num_books",
    "first_author_num_followers",
    "average_rating",
    "num_reviews",
    "num_pages",
    "num_currently_reading",
    "num_want_to_read",
)


@dataclass(slots=True)
class BookRecord:
    """A normalized book record ready for transformation and persistence.

    Attributes:
        book_id: Unique string identifier for the book on Goodreads.
        url: Full canonical web URL to the book's detail page.
        title: Title of the book.
        first_author: Primary author's name.
        first_author_url: Full web URL to the primary author's Goodreads profile.
        first_author_num_books: Total number of published works by the primary author.
                            Defaults to 0.
        first_author_num_followers: Total Goodreads follower count for the primary author.
                                Defaults to 0.
        average_rating: Overall average star rating (0.0 to 5.0).
                    Defaults to 0.0.
        num_reviews: Total user rating/review count.
                    Defaults to 0.
        num_pages: Page count of the specific edition.
                Defaults to 0.
        num_currently_reading: Count of Goodreads users currently reading this book.
                            Defaults to 0.
        num_want_to_read: Count of Goodreads users who marked this book as want-to-read.
                        Defaults to 0.
        first_published: Date string of initial publication (e.g. 'October 1, 2020').
                        Defaults to None.
        publisher: Name of the publishing company.
                Defaults to None.
        language_code: ISO or standard language string (e.g. 'eng', 'en-US').
                    Defaults to None.
        description: Full plain text book synopsis/blurb.
                    Defaults to None.
        genres: Delimited list or formatted string of associated genres.
            Defaults to None.
        format: Publication format type (e.g. 'Hardcover', 'Paperback', 'Ebook').
            Defaults to None.
        series: Name of the literary series, if applicable.
            Defaults to None.
    """

    book_id: str
    url: str
    title: str
    first_author: str
    first_author_url: str

    first_author_num_books: int = 0
    first_author_num_followers: int = 0
    average_rating: float = 0.0
    num_reviews: int = 0
    num_pages: int = 0
    num_currently_reading: int = 0
    num_want_to_read: int = 0

    first_published: str | None = None
    publisher: str | None = None
    language_code: str | None = None
    description: str | None = None
    genres: str | None = None
    format: str | None = None
    series: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the record as a plain dictionary, column-name keyed.

        Returns:
            A dictionary mapping each dataclass field name to its corresponding value.
        """
        result: dict[str, object] = asdict(self)
        return result
