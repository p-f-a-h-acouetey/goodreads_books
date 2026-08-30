"""Pytest suite for extract.py's BookExtractor.

Split into groups matching BookExtractor's own internal structure:
- Parsing (pure functions fed real-shaped mock HTML via BeautifulSoup)
- Record assembly (pure dict manipulation)
- Request building (real crawlee.Request objects, no network)
- _Progress (the nested dataclass, pure state machine)
- Crawl construction and routing (_build_crawler, _attach_handlers' nested
  route/failure handlers, _drive_to_completion, _reset_request_queue_storage)
- extract() orchestration (the async entry point, with the crawler itself
  mocked out since a real crawl requires network access)

No network calls are made anywhere in this suite. Crawlee's
BeautifulSoupCrawler is only ever constructed (never .run() against a real
queue) except where explicitly faked; extract()'s own orchestration logic
is tested by patching _build_crawler/_attach_handlers/_drive_to_completion.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger
from bs4 import BeautifulSoup
from crawlee import Request

import src.goodreads_etl.extractors.extract as extract_module
from src.goodreads_etl.extractors.extract import BookExtractor, EDITIONS_URL_TEMPLATE, BASE_BOOK_URL

_RAISES = object()


# ---------------------------------------------------------------------------
# Mock HTML builders
# ---------------------------------------------------------------------------


def _build_book_html(
    book_id: str = "123",
    title: str | None = "Some Book",
    include_work: bool = True,
    catalog_violation: bool = False,
    publication_time_ms: int | None = -3471292800000,  # ~1860, pre-1970 dates
) -> str:
    """Build mock HTML for a Goodreads book page with a realistic __NEXT_DATA__ payload."""
    body_text = "This book does not meet our catalog guidelines." if catalog_violation else ""

    book_key = "Book:kca://book/x"
    work_key = "Work:1"
    contributor_key = "Contributor:1"
    contributor2_key = "Contributor:2"

    apollo_state: dict = {
        book_key: {
            "legacyId": book_id,
            "title": title,
            "description": "A great book.",
            "details": {
                "format": "Hardcover",
                "numPages": 300,
                "language": {"name": "English"},
                "asin": "1234567890",
                "isbn": "1234567890",
                "isbn13": "9781234567890",
                "publisher": "Test Publisher",
                "publicationTime": publication_time_ms,
            },
            "bookGenres": [{"genre": {"name": "Fiction"}}, {"genre": {"name": "Drama"}}],
            "bookSeries": [{"series": {"title": "Great Series"}, "userPosition": "1"}],
            "work": {"__ref": work_key} if include_work else None,
            "primaryContributorEdge": {"role": "Author", "node": {"__ref": contributor_key}},
            "secondaryContributorEdges": [{"role": "Editor", "node": {"__ref": contributor2_key}}],
        }
    }
    if include_work:
        apollo_state[work_key] = {
            "legacyId": 999,
            "stats": {"averageRating": 4.2, "ratingsCount": 1000, "textReviewsCount": 50},
        }
    apollo_state[contributor_key] = {"name": "Jane Author", "webUrl": "https://goodreads.com/author/1"}
    apollo_state[contributor2_key] = {"name": "John Editor", "webUrl": "https://goodreads.com/author/2"}

    next_data = {"props": {"pageProps": {"apolloState": apollo_state}}}
    return f'<html><body><script id="__NEXT_DATA__">{json.dumps(next_data)}</script>{body_text}</body></html>'


def _soup(html: str) -> BeautifulSoup:
    """Parse HTML into a BeautifulSoup object, mirroring what Crawlee hands to handlers."""
    return BeautifulSoup(html, "html.parser")


def _context(request: Request, soup: BeautifulSoup | None = None) -> SimpleNamespace:
    """Build a minimal stand-in for BeautifulSoupCrawlingContext.

    Real Crawlee contexts are plain data holders exposing .request, .soup,
    and an async .add_requests(...) method; a SimpleNamespace with an
    AsyncMock for add_requests is enough to drive route handlers directly,
    without depending on unverified Crawlee internals.
    """
    return SimpleNamespace(request=request, soup=soup, add_requests=AsyncMock())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor():
    """A plain BookExtractor instance -- __init__ has no side effects or I/O."""
    return BookExtractor()


# ---------------------------------------------------------------------------
# GROUP 1 -- Parsing
# ---------------------------------------------------------------------------


class TestMsEpochToIso:
    """Tests for _ms_epoch_to_iso, including the Windows pre-1970 regression."""

    @pytest.mark.parametrize(
        "ms, expected_iso",
        [
            (None, None),  # falsy input
            (0, None),  # falsy input
            (int(datetime(2023, 11, 14, tzinfo=timezone.utc).timestamp() * 1000), "2023-11-14"),  # normal post-1970 date
            (-3471292800000, "1860-01-01"),  # pre-1970 regression: Windows OSError case
        ],
    )
    def test_converts_date_correctly(self, extractor, ms, expected_iso):
        """A millisecond-epoch timestamp should convert to the correct ISO date, or None if falsy.

        Covers falsy input (None/0), a normal post-1970 date, and a
        pre-1970 date (regression: datetime.fromtimestamp() raises
        OSError: [Errno 22] Invalid argument on Windows for any date
        before 1970-01-02 -- _ms_epoch_to_iso must use timedelta
        arithmetic instead).
        """
        assert extractor._ms_epoch_to_iso(ms) == expected_iso


class TestResolveRefs:
    """Tests for _resolve_refs, the Apollo cache __ref de-referencer."""

    @pytest.mark.parametrize(
        "state, node, expected",
        [
            ({"Work:1": {"title": "Target"}}, {"__ref": "Work:1"}, {"title": "Target"}),
            (
                {"Work:1": {"title": "Target"}},
                {"work": {"__ref": "Work:1"}, "other": "value"},
                {"work": {"title": "Target"}, "other": "value"},
            ),
            (
                {"A:1": {"name": "First"}, "A:2": {"name": "Second"}},
                [{"__ref": "A:1"}, {"__ref": "A:2"}],
                [{"name": "First"}, {"name": "Second"}],
            ),
            ({}, "hello", "hello"),
            ({}, 42, 42),
            ({}, None, None),
            ({}, {"__ref": "Missing:1"}, None),
        ],
    )
    def test_resolve_refs_cases(self, extractor, state, node, expected):
        """_resolve_refs should correctly resolve simple/nested/list refs, pass through scalars
        unchanged, and resolve a missing ref target to None.

        Covers: a single __ref pointer, a __ref nested inside a larger
        dict, __ref pointers inside a list, non-dict/list scalars passed
        through unchanged (str/int/None), and a __ref pointing at a key
        absent from state.
        """
        assert extractor._resolve_refs(state, node) == expected

    def test_returns_none_past_max_depth(self, extractor, monkeypatch):
        """Recursion past APOLLO_CACHE_MAX_DEPTH should return None, not recurse forever."""
        monkeypatch.setattr(extract_module, "APOLLO_CACHE_MAX_DEPTH", 2)
        state = {"A:1": {"__ref": "A:2"}, "A:2": {"__ref": "A:3"}, "A:3": {"value": "too deep"}}
        result = extractor._resolve_refs(state, {"__ref": "A:1"})
        assert result is None


class TestLoadApolloState:
    """Tests for _load_apollo_state."""

    def test_loads_valid_next_data(self, extractor):
        """A well-formed __NEXT_DATA__ script tag should yield the apolloState dict."""
        html = _build_book_html()
        state = extractor._load_apollo_state(_soup(html))
        assert state is not None
        assert "Book:kca://book/x" in state

    @pytest.mark.parametrize(
        "html",
        [
            "<html><body>No data here</body></html>",
            '<html><body><script id="__NEXT_DATA__"></script></body></html>',
            '<html><body><script id="__NEXT_DATA__">{not valid json</script></body></html>',
            '<html><body><script id="__NEXT_DATA__">{"unexpected": "shape"}</script></body></html>',
        ],
        ids=["missing_tag", "empty_tag", "malformed_json", "missing_expected_keys"],
    )
    def test_returns_none_for_invalid_input(self, extractor, html):
        """_load_apollo_state should return None, not raise, for any malformed input.

        Covers: no __NEXT_DATA__ tag at all, an empty tag, invalid JSON
        inside the tag, and valid JSON missing the expected
        props/pageProps/apolloState chain.
        """
        assert extractor._load_apollo_state(_soup(html)) is None


class TestFindBookEntry:
    """Tests for _find_book_entry."""

    @pytest.mark.parametrize(
        "state, book_id, expected_title",
        [
            ({"Book:x": {"legacyId": "123", "title": "Found Me"}}, "123", "Found Me"),
            ({"Book:x": {"legacyId": "999", "title": "Wrong Book"}}, "123", None),
            (
                {
                    "Book:a": {"legacyId": "111", "title": "Other Book"},
                    "Book:b": {"legacyId": "123", "title": "Target Book"},
                },
                "123",
                "Target Book",
            ),
            ({"Book:x": {"legacyId": 123, "title": "Found"}}, "123", "Found"),
        ],
        ids=["matching_id", "no_matching_id", "ignores_other_books", "matches_int_or_str_id"],
    )
    def test_find_book_entry_cases(self, extractor, state, book_id, expected_title):
        """_find_book_entry should locate the Book: entry matching book_id, or return None.

        Covers: a direct legacyId match, no match found, correctly
        ignoring unrelated Book: entries in the same cache (e.g. "readers
        also enjoyed" widgets), and matching regardless of whether
        legacyId is stored as a string or an int.
        """
        result = extractor._find_book_entry(state, book_id)
        if expected_title is None:
            assert result is None
        else:
            assert result["title"] == expected_title


class TestExtractAllContributors:
    """Tests for _extract_all_contributors."""

    @pytest.mark.parametrize(
        "book, expected",
        [
            (
                {"primaryContributorEdge": {"role": "Author", "node": {"name": "Solo Author", "webUrl": "u1"}}},
                [{"name": "Solo Author", "url": "u1", "role": "Author"}],
            ),
            (
                {
                    "primaryContributorEdge": {"role": "Author", "node": {"name": "Main", "webUrl": "u1"}},
                    "secondaryContributorEdges": [{"role": "Illustrator", "node": {"name": "Artist", "webUrl": "u2"}}],
                },
                [
                    {"name": "Main", "url": "u1", "role": "Author"},
                    {"name": "Artist", "url": "u2", "role": "Illustrator"},
                ],
            ),
            (
                {"primaryContributorEdge": {"node": {"name": "X", "webUrl": "u"}}},
                [{"name": "X", "url": "u", "role": "Author"}],
            ),
            ({"secondaryContributorEdges": [{"node": {"webUrl": "u"}}]}, []),
            (
                {"secondaryContributorEdges": [None, "not a dict", 42, {"role": "Illustrator", "node": {"name": "Real", "webUrl": "u"}}]},
                [{"name": "Real", "url": "u", "role": "Illustrator"}],
            ),
            ({}, []),
        ],
        ids=[
            "primary_only",
            "primary_and_secondary",
            "defaults_role_to_author",
            "skips_missing_name",
            "skips_non_dict_secondary_edges",
            "no_contributors",
        ],
    )
    def test_extract_all_contributors_cases(self, extractor, book, expected):
        """_extract_all_contributors should extract primary/secondary contributors correctly.

        Covers: a primary-only contributor, primary plus secondary
        (primary first), a primary edge with no explicit role defaulting
        to "Author", a secondary entry with no name being skipped, a
        non-dict secondary edge being skipped without raising, and a book
        with no contributors at all yielding an empty list.
        """
        assert extractor._extract_all_contributors(book) == expected


class TestParseBookPage:
    """Tests for _parse_book_page, the main Apollo-cache book parser."""

    def test_parses_full_valid_book(self, extractor):
        """A well-formed book page should populate every expected field."""
        html = _build_book_html(book_id="123", title="Great Book")
        data = extractor._parse_book_page(_soup(html), "123")
        assert data["title"] == "Great Book"
        assert data["is_nonexistent"] is False
        assert data["first_author"] == "Jane Author"
        assert data["first_author_url"] == "https://goodreads.com/author/1"
        assert data["genres"] == ["Fiction", "Drama"]
        assert data["series"] == [{"name": "Great Series", "position": "1"}]
        assert data["asin"] == "1234567890"
        assert data["isbn10"] == "1234567890"
        assert data["isbn13"] == "9781234567890"
        assert data["publisher"] == "Test Publisher"
        assert data["average_rating"] == 4.2
        assert data["ratings_count"] == 1000
        assert data["reviews_count"] == 50
        assert data["work_id"] == 999
        assert data["editions_url"] == EDITIONS_URL_TEMPLATE.format(work_id=999)
        assert data["description"] == "A great book."

    def test_no_apollo_state_treated_as_nonexistent(self, extractor):
        """A page with no __NEXT_DATA__ at all should be treated as non-existent."""
        html = "<html><body>Nothing here</body></html>"
        data = extractor._parse_book_page(_soup(html), "123")
        assert data["is_nonexistent"] is True

    def test_missing_title_treated_as_nonexistent(self, extractor):
        """A Book entry present but with no title should be treated as non-existent."""
        html = _build_book_html(title="")
        data = extractor._parse_book_page(_soup(html), "123")
        assert data["is_nonexistent"] is True
        assert data["title"] is None

    def test_catalog_guideline_violation_is_not_nonexistent(self, extractor):
        """A catalog-guideline-violation page should be flagged but NOT treated as non-existent."""
        html = _build_book_html(title=None, catalog_violation=True)
        data = extractor._parse_book_page(_soup(html), "123")
        assert data["catalog_guideline_violation"] is True
        assert data["is_nonexistent"] is False

    def test_editions_url_none_when_no_work_id(self, extractor):
        """A book with no resolvable work_id should have editions_url=None."""
        html = _build_book_html(include_work=False)
        data = extractor._parse_book_page(_soup(html), "123")
        assert data["editions_url"] is None
        assert data["work_id"] is None


class TestParseAuthorPage:
    """Tests for _parse_author_page."""

    @pytest.mark.parametrize(
        "html, expected",
        [
            (
                "<html><body>Followers (1,234) 4,586 some text 56 distinct works</body></html>",
                {"num_followers": 1234, "num_distinct_works": 56},
            ),
            (
                "<html><body>1,200,563 works and 42 distinct works</body></html>",
                {"num_followers": None, "num_distinct_works": 42},
            ),
            (
                "<html><body>Nothing relevant here : 4,500 followers and 1,200 works</body></html>",
                {"num_followers": None, "num_distinct_works": None},
            ),
        ],
        ids=["extracts_followers_and_works", "missing_followers", "missing_both"],
    )
    def test_parse_author_page_cases(self, extractor, html, expected):
        """_parse_author_page should extract followers/distinct works counts, or None if absent.

        Covers: both fields present and parsed correctly, followers
        missing while distinct works is still found, and neither pattern
        present in the page at all.
        """
        result = extractor._parse_author_page(_soup(html))
        assert result == expected


class TestParseEditionsPage:
    """Tests for _parse_editions_page."""

    @pytest.mark.parametrize(
        "html, expected",
        [
            ("<html><body>Showing 1-10 of 1,200</body></html>", 1200),
            ("<html><body>Showing all 5</body></html>", 5),
            ("<html><body>2 Editions</body></html>", 2),
            ("<html><body>No relevant text</body></html>", None),
        ],
        ids=[
            "showing_x_of_y_pattern",
            "showing_all_pattern",
            "falls_back_to_editions_heading",
            "no_pattern_matches",
        ],
    )
    def test_parse_editions_page_cases(self, extractor, html, expected):
        """_parse_editions_page should extract the total edition count, or None if absent.

        Covers: the "Showing X-Y of TOTAL" pagination pattern, the
        "Showing all N" pattern, falling back to a bare "N Editions"
        heading when no pagination text is present, and returning None
        when neither pattern matches.
        """
        assert extractor._parse_editions_page(_soup(html)) == expected


# ---------------------------------------------------------------------------
# GROUP 2 -- Record assembly
# ---------------------------------------------------------------------------


class TestStartRecord:
    """Tests for _start_record."""

    def test_builds_correct_initial_shape(self, extractor):
        """The initial pending record should include book_id, url, and null enrichment fields."""
        book_data = {"title": "X", "first_author_url": "u", "editions_url": "e"}
        record = extractor._start_record("123", book_data)
        assert record["book_id"] == "123"
        assert record["url"] == f"{BASE_BOOK_URL}123"
        assert record["num_editions"] is None
        assert record["has_more_editions"] is False

    def test_sets_awaiting_flags_when_urls_present(self, extractor):
        """Both _awaiting flags should be True when both enrichment URLs exist."""
        book_data = {"first_author_url": "u", "editions_url": "e"}
        record = extractor._start_record("123", book_data)
        assert record["_awaiting_author"] is True
        assert record["_awaiting_editions"] is True

    def test_awaiting_flags_false_when_urls_absent(self, extractor):
        """Both _awaiting flags should be False when neither enrichment URL exists."""
        book_data = {"first_author_url": None, "editions_url": None}
        record = extractor._start_record("123", book_data)
        assert record["_awaiting_author"] is False
        assert record["_awaiting_editions"] is False

    def test_strips_internal_only_fields(self, extractor):
        """is_nonexistent, first_author_url, and editions_url should not leak into the record."""
        book_data = {"is_nonexistent": False, "first_author_url": "u", "editions_url": "e", "title": "X"}
        record = extractor._start_record("123", book_data)
        assert "is_nonexistent" not in record
        assert "first_author_url" not in record
        assert "editions_url" not in record
        assert record["title"] == "X"


class TestIsReadyToAssemble:
    """Tests for _is_ready_to_assemble."""

    @pytest.mark.parametrize(
        "record, expected",
        [
            ({"_awaiting_author": False, "_awaiting_editions": False}, True),
            ({"_awaiting_author": True, "_awaiting_editions": False}, False),
            ({"_awaiting_author": False, "_awaiting_editions": True}, False),
            ({"_awaiting_author": True, "_awaiting_editions": True}, False),
        ],
        ids=["neither_awaited", "awaiting_author", "awaiting_editions", "awaiting_both"],
    )
    def test_is_ready_to_assemble_cases(self, extractor, record, expected):
        """_is_ready_to_assemble should be True only when neither awaiting flag is set.

        Covers all four combinations of _awaiting_author / _awaiting_editions.
        """
        assert extractor._is_ready_to_assemble(record) is expected


class TestAssembleRecord:
    """Tests for _assemble_record."""

    @pytest.mark.parametrize(
        "record, expected_is_part_of_series",
        [
            ({"_awaiting_author": False, "_awaiting_editions": False, "title": "X", "series": []}, False),
            (
                {"_awaiting_author": False, "_awaiting_editions": False, "title": "Y", "series": [{"name": "S", "position": "1"}]},
                True,
            ),
        ],
        ids=["strips_flags_no_series", "series_present"],
    )
    def test_assemble_record_cases(self, extractor, record, expected_is_part_of_series):
        """_assemble_record should strip internal _awaiting_* flags and set is_part_of_series correctly.

        Covers: a record with both awaiting flags present and an empty
        series (flags stripped, is_part_of_series False), and a record
        with a non-empty series list (is_part_of_series True).
        """
        result = extractor._assemble_record(record)
        assert "_awaiting_author" not in result
        assert "_awaiting_editions" not in result
        assert result["is_part_of_series"] is expected_is_part_of_series


# ---------------------------------------------------------------------------
# GROUP 3 -- Request building
# ---------------------------------------------------------------------------


class TestRequestBuilding:
    """Tests for _build_book_request / _build_author_request / _build_editions_request."""

    def test_build_book_request(self, extractor):
        """A book request should target the right URL, label, and user_data."""
        req = extractor._build_book_request("123")
        assert req.url == f"{BASE_BOOK_URL}123"
        assert req.label == "book"
        assert req.user_data["book_id"] == "123"

    def test_build_author_request(self, extractor):
        """An author request should target the given author URL with book_id attached."""
        req = extractor._build_author_request("https://goodreads.com/author/1", "123")
        assert req.url == "https://goodreads.com/author/1"
        assert req.label == "author"
        assert req.user_data["book_id"] == "123"

    def test_build_editions_request(self, extractor):
        """An editions request should target the given editions URL with book_id attached."""
        req = extractor._build_editions_request("https://goodreads.com/work/editions/999", "123")
        assert req.url == "https://goodreads.com/work/editions/999"
        assert req.label == "editions"
        assert req.user_data["book_id"] == "123"


class TestRequireBookId:
    """Tests for _require_book_id."""

    @pytest.mark.parametrize(
        "user_data, expected",
        [
            ({"book_id": "123"}, "123"),
            ({}, _RAISES),
            ({"book_id": 123}, _RAISES),
        ],
        ids=["valid_book_id", "missing_book_id", "wrong_type_book_id"],
    )
    def test_require_book_id_cases(self, extractor, user_data, expected):
        """_require_book_id should return a valid string book_id, or raise ValueError otherwise.

        Covers: a valid string book_id returned as-is, book_id missing
        from user_data, and book_id present but of the wrong type (int
        instead of str).
        """
        req = Request.from_url("http://x", user_data=user_data)
        if expected is _RAISES:
            with pytest.raises(ValueError, match="Missing or invalid book_id"):
                extractor._require_book_id(req)
        else:
            assert extractor._require_book_id(req) == expected


class TestOptionalBookId:
    """Tests for _optional_book_id."""

    @pytest.mark.parametrize(
        "user_data, expected",
        [
            ({"book_id": "123"}, "123"),
            ({}, None),
            ({"book_id": 123}, None),
        ],
        ids=["valid_book_id", "missing_book_id", "wrong_type_book_id"],
    )
    def test_optional_book_id_cases(self, extractor, user_data, expected):
        """_optional_book_id should return a valid string book_id, or None otherwise.

        Covers: a valid string book_id returned as-is, book_id missing
        from user_data, and book_id present but of the wrong type (int
        instead of str) -- neither invalid case should raise.
        """
        req = Request.from_url("http://x", user_data=user_data)
        assert extractor._optional_book_id(req) == expected


# ---------------------------------------------------------------------------
# GROUP 4 -- _Progress
# ---------------------------------------------------------------------------


class TestProgress:
    """Tests for BookExtractor._Progress, the per-run crawl state dataclass."""

    def test_successes_and_is_complete(self):
        """successes should count records; is_complete should trigger at target_count."""
        progress = BookExtractor._Progress(target_count=2, tried_book_ids=set())
        assert progress.successes == 0
        assert progress.is_complete is False
        progress.records.append({"book_id": "1"})
        assert progress.successes == 1
        assert progress.is_complete is False
        progress.records.append({"book_id": "2"})
        assert progress.is_complete is True

    def test_next_random_book_id_avoids_tried_ids(self, monkeypatch):
        """next_random_book_id should never return an id already in tried_book_ids."""
        progress = BookExtractor._Progress(target_count=1, tried_book_ids={"1", "2"})
        call_sequence = iter(["1", "2", "3"])
        monkeypatch.setattr(extract_module.random, "randint", lambda *_: int(next(call_sequence)))
        result = progress.next_random_book_id()
        assert result == "3"
        assert "3" in progress.tried_book_ids

    def test_next_random_book_id_gives_up_after_max_attempts(self, monkeypatch):
        """If every draw collides, next_random_book_id should return None, not loop forever."""
        monkeypatch.setattr(extract_module, "MAX_RANDOM_ID_ATTEMPTS", 5)
        progress = BookExtractor._Progress(target_count=1, tried_book_ids={"1"})
        monkeypatch.setattr(extract_module.random, "randint", lambda *_: 1)
        assert progress.next_random_book_id() is None

    def test_add_record_rejects_once_complete(self):
        """add_record should return False and not append once the target is already met."""
        progress = BookExtractor._Progress(target_count=1, tried_book_ids=set())
        assert progress.add_record({"book_id": "1"}) is True
        assert progress.add_record({"book_id": "2"}) is False
        assert progress.successes == 1

    def test_add_record_sets_done_event_when_target_met(self):
        """done_event should be set exactly when the target_count is reached."""
        progress = BookExtractor._Progress(target_count=1, tried_book_ids=set())
        assert not progress.done_event.is_set()
        progress.add_record({"book_id": "1"})
        assert progress.done_event.is_set()


# ---------------------------------------------------------------------------
# GROUP 5 -- Crawl construction and routing
#
# _build_crawler is a pure constructor, tested directly. _attach_handlers
# registers four nested closures (handle_book/handle_author/handle_editions
# via @crawler.router.handler(label), plus handle_failed via
# @crawler.failed_request_handler) that are otherwise unreachable by name --
# a minimal recording double (_FakeRouter/_FakeCrawler) is used purely to
# grab a direct handle on them, exactly as _attach_handlers itself expects
# to wire against: an object exposing .router.handler(label) and
# .failed_request_handler as decorators. Each handler is then called with a
# SimpleNamespace context (a plain stand-in for Crawlee's own plain context
# object) and real crawlee.Request instances -- no MagicMock behavior is
# relied upon anywhere in this group.
# ---------------------------------------------------------------------------


class _FakeRouter:
    """Records handlers registered via @crawler.router.handler(label)."""

    def __init__(self):
        self.handlers: dict[str, Any] = {}

    def handler(self, label):
        def _decorator(fn):
            self.handlers[label] = fn
            return fn

        return _decorator


class _FakeCrawler:
    """Minimal double exposing just the two attachment points _attach_handlers needs."""

    def __init__(self):
        self.router = _FakeRouter()
        self.failed_handler = None

    def failed_request_handler(self, fn):
        self.failed_handler = fn
        return fn


class TestBuildCrawler:
    """Tests for _build_crawler."""

    def test_returns_beautifulsoup_crawler_instance(self, extractor):
        """_build_crawler should return an un-routed BeautifulSoupCrawler instance."""
        from crawlee.crawlers import BeautifulSoupCrawler

        crawler = extractor._build_crawler()
        assert isinstance(crawler, BeautifulSoupCrawler)


class TestAttachHandlersRouting:
    """Structural tests confirming _attach_handlers wires up all expected routes."""

    def test_registers_book_author_editions_and_failed_handlers(self, extractor):
        """All three page-label handlers and the failed-request handler should be registered."""
        crawler = _FakeCrawler()
        progress = BookExtractor._Progress(target_count=1, tried_book_ids=set())
        extractor._attach_handlers(crawler, progress)

        assert set(crawler.router.handlers) == {
            extract_module.LABEL_BOOK,
            extract_module.LABEL_AUTHOR,
            extract_module.LABEL_EDITIONS,
        }
        assert crawler.failed_handler is not None


class TestEnqueueReplacement:
    """Tests every branch of _enqueue_replacement through handle_book's
    nonexistent-book path.
    """

    @pytest.fixture
    def progress(self):
        return BookExtractor._Progress(target_count=1, tried_book_ids=set())

    @pytest.fixture
    def handle_book(self, extractor, progress):
        crawler = _FakeCrawler()
        extractor._attach_handlers(crawler, progress)
        return crawler.router.handlers[extract_module.LABEL_BOOK]

    @pytest.fixture
    def nonexistent_book(self, extractor, monkeypatch):
        """Make handle_book always take its nonexistent-book path."""
        monkeypatch.setattr(extractor, "_parse_book_page", lambda soup, book_id: {"is_nonexistent": True})

    async def test_returns_none_when_complete(self, progress, handle_book, nonexistent_book, monkeypatch):
        """If progress is complete, _enqueue_replacement should return at its first line --
        next_random_book_id must never be called, and no request should be enqueued.
        """
        progress.add_record({"book_id": "1"})
        assert progress.is_complete is True

        next_id = MagicMock()
        monkeypatch.setattr(BookExtractor._Progress, "next_random_book_id", lambda self: next_id())

        context = _context(Request.from_url("http://x", label=extract_module.LABEL_BOOK, user_data={"book_id": "2"}))
        result = await handle_book(context)

        assert result is None
        next_id.assert_not_called()
        context.add_requests.assert_not_awaited()

    async def test_returns_none_when_no_id_is_available(self, handle_book, nonexistent_book, monkeypatch):
        """If no untried ID exists, _enqueue_replacement should log a warning and skip enqueuing."""
        next_id = MagicMock(return_value=None)
        monkeypatch.setattr(BookExtractor._Progress, "next_random_book_id", lambda self: next_id())

        context = _context(
            Request.from_url("http://x", label=extract_module.LABEL_BOOK, user_data={"book_id": "123"})
        )

        messages = []
        sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
        try:
            result = await handle_book(context)
        finally:
            logger.remove(sink_id)

        assert result is None
        next_id.assert_called_once()
        context.add_requests.assert_not_awaited()
        assert any("No untried book ID found" in m for m in messages)

    async def test_returns_none_after_enqueuing_replacement(self, extractor, handle_book, nonexistent_book, monkeypatch):
        """If an ID is available, _enqueue_replacement should build and enqueue exactly one
        book request for that ID, then return None.
        """
        next_id = MagicMock(return_value="999")
        build_request = MagicMock(
            return_value=Request.from_url("http://x/book/999", label=extract_module.LABEL_BOOK, user_data={"book_id": "999"})
        )
        monkeypatch.setattr(BookExtractor._Progress, "next_random_book_id", lambda self: next_id())
        monkeypatch.setattr(extractor, "_build_book_request", build_request)

        context = _context(
            Request.from_url("http://x", label=extract_module.LABEL_BOOK, user_data={"book_id": "123"})
        )

        result = await handle_book(context)

        assert result is None
        next_id.assert_called_once()
        build_request.assert_called_once_with("999")
        context.add_requests.assert_awaited_once_with([build_request.return_value])

    async def test_book_failure_path_also_respects_is_complete(self, extractor, monkeypatch):
        """_enqueue_replacement is also called from handle_failed's book-failure path --
        confirm that entry point respects the is_complete guard too, not just handle_book's.
        """
        crawler = _FakeCrawler()
        progress = BookExtractor._Progress(target_count=1, tried_book_ids=set())
        extractor._attach_handlers(crawler, progress)
        handle_failed = crawler.failed_handler

        progress.add_record({"book_id": "already-done"})
        assert progress.is_complete is True

        next_id = MagicMock()
        monkeypatch.setattr(BookExtractor._Progress, "next_random_book_id", lambda self: next_id())

        context = _context(
            Request.from_url("http://x", label=extract_module.LABEL_BOOK, user_data={"book_id": "123"})
        )
        result = await handle_failed(context, Exception("boom")) # type: ignore

        assert result is None
        next_id.assert_not_called()
        context.add_requests.assert_not_awaited()


class TestTryFinalize:
    """Tests every branch of _try_finalize through handle_editions, the simplest
    handler that calls it with a directly controllable pending record.
    """

    @pytest.fixture
    def progress(self):
        return BookExtractor._Progress(target_count=3, tried_book_ids=set())

    @pytest.fixture
    def handle_editions(self, extractor, progress):
        crawler = _FakeCrawler()
        extractor._attach_handlers(crawler, progress)
        return crawler.router.handlers[extract_module.LABEL_EDITIONS]

    async def test_returns_none_when_no_pending_record(self, extractor, progress, handle_editions, monkeypatch):
        """If record is None, _try_finalize should return immediately -- no assembly, no
        add_record, no logging, and no enqueue attempt.
        """
        assemble = MagicMock()
        monkeypatch.setattr(extractor, "_assemble_record", assemble)
        monkeypatch.setattr(extractor, "_parse_editions_page", lambda soup: 5)
        # Deliberately no progress.pending["123"] entry.

        context = _context(
            Request.from_url("http://editions", label=extract_module.LABEL_EDITIONS, user_data={"book_id": "123"})
        )
        result = await handle_editions(context)

        assert result is None
        assemble.assert_not_called()
        assert progress.records == []
        context.add_requests.assert_not_awaited()

    async def test_returns_none_when_not_ready_to_assemble(self, extractor, progress, handle_editions, monkeypatch):
        """If the record still has an outstanding enrichment flag, _try_finalize should return
        without popping it from pending or assembling it.
        """
        progress.pending["123"] = {
            "contributors": [],
            "_awaiting_author": True,  # still outstanding -- handle_editions only clears editions
            "_awaiting_editions": True,
        }
        assemble = MagicMock()
        monkeypatch.setattr(extractor, "_assemble_record", assemble)
        monkeypatch.setattr(extractor, "_parse_editions_page", lambda soup: 5)

        context = _context(
            Request.from_url("http://editions", label=extract_module.LABEL_EDITIONS, user_data={"book_id": "123"})
        )
        result = await handle_editions(context)

        assert result is None
        assert "123" in progress.pending  # NOT popped
        assemble.assert_not_called()
        assert progress.records == []
        context.add_requests.assert_not_awaited()

    async def test_returns_none_when_add_record_rejects(self, extractor, progress, handle_editions, monkeypatch):
        """If add_record() returns False (run already complete via a different concurrent book),
        _try_finalize should return without logging success or enqueuing a replacement -- even
        though the record was already popped and assembled.
        """
        progress.pending["123"] = {"contributors": [], "_awaiting_author": False, "_awaiting_editions": True}
        monkeypatch.setattr(BookExtractor._Progress, "add_record", lambda self, record: False)
        monkeypatch.setattr(extractor, "_parse_editions_page", lambda soup: 5)

        context = _context(
            Request.from_url("http://editions", label=extract_module.LABEL_EDITIONS, user_data={"book_id": "123"})
        )

        messages = []
        sink_id = logger.add(lambda msg: messages.append(msg), level="INFO")
        try:
            result = await handle_editions(context)
        finally:
            logger.remove(sink_id)

        assert result is None
        assert "123" not in progress.pending  # already popped before add_record was attempted
        assert progress.records == []  # never stored, since add_record returned False
        context.add_requests.assert_not_awaited()
        assert not any("collected" in m for m in messages)  # success log never reached

    async def test_completes_run_skips_replacement(self, extractor, handle_editions, monkeypatch):
        """If this record completes the run (is_complete becomes True), the success log should
        fire, but _enqueue_replacement should NOT be called.
        """
        progress = BookExtractor._Progress(target_count=1, tried_book_ids=set())
        crawler = _FakeCrawler()
        extractor._attach_handlers(crawler, progress)
        handle_editions = crawler.router.handlers[extract_module.LABEL_EDITIONS]

        progress.pending["123"] = {"contributors": [], "_awaiting_author": False, "_awaiting_editions": True}
        monkeypatch.setattr(extractor, "_parse_editions_page", lambda soup: 5)

        context = _context(
            Request.from_url("http://editions", label=extract_module.LABEL_EDITIONS, user_data={"book_id": "123"})
        )

        messages = []
        sink_id = logger.add(lambda msg: messages.append(msg), level="INFO")
        try:
            result = await handle_editions(context)
        finally:
            logger.remove(sink_id)

        assert result is None
        assert progress.successes == 1
        assert progress.is_complete is True
        assert any("book_id=123" in m and "1/1" in m for m in messages)
        context.add_requests.assert_not_awaited()

    async def test_incomplete_run_triggers_replacement(self, extractor, progress, handle_editions, monkeypatch):
        """If the run is NOT yet complete after this record, _try_finalize should both log
        success and enqueue a replacement via _enqueue_replacement.
        """
        progress.pending["123"] = {"contributors": [], "_awaiting_author": False, "_awaiting_editions": True}
        monkeypatch.setattr(extractor, "_parse_editions_page", lambda soup: 5)
        monkeypatch.setattr(BookExtractor._Progress, "next_random_book_id", lambda self: "999")

        context = _context(
            Request.from_url("http://editions", label=extract_module.LABEL_EDITIONS, user_data={"book_id": "123"})
        )

        messages = []
        sink_id = logger.add(lambda msg: messages.append(msg), level="INFO")
        try:
            result = await handle_editions(context)
        finally:
            logger.remove(sink_id)

        assert result is None
        assert progress.successes == 1
        assert progress.is_complete is False
        assert any("book_id=123" in m and "1/3" in m for m in messages)
        context.add_requests.assert_awaited_once()
        (enqueued_requests,), _ = context.add_requests.call_args
        assert enqueued_requests[0].user_data["book_id"] == "999"


class TestHandleBook:
    """Tests for the handle_book route handler wired up by _attach_handlers."""

    @pytest.fixture
    def progress(self):
        """A fresh _Progress with target_count=3, so a single finalize never completes the run."""
        return BookExtractor._Progress(target_count=3, tried_book_ids=set())

    @pytest.fixture
    def handle_book(self, extractor, progress):
        """The handle_book closure, extracted from a real _attach_handlers call."""
        crawler = _FakeCrawler()
        extractor._attach_handlers(crawler, progress)
        return crawler.router.handlers[extract_module.LABEL_BOOK]

    async def test_nonexistent_book_enqueues_replacement_not_pending_record(
        self, extractor, progress, handle_book, monkeypatch
    ):
        """A book page parsed as nonexistent should enqueue a replacement, not create a pending record."""
        monkeypatch.setattr(extractor, "_parse_book_page", lambda soup, book_id: {"is_nonexistent": True})
        monkeypatch.setattr(BookExtractor._Progress, "next_random_book_id", lambda self: "999")

        context = _context(Request.from_url("http://x", label="book", user_data={"book_id": "123"}))
        await handle_book(context)

        assert "123" not in progress.pending
        context.add_requests.assert_awaited_once()

    async def test_done_event_already_set_skips_everything(self, extractor, progress, handle_book):
        """If progress.done_event is already set, handle_book should do nothing at all."""
        progress.done_event.set()

        context = _context(Request.from_url("http://x", label="book", user_data={"book_id": "123"}))
        await handle_book(context)

        assert progress.pending == {}
        context.add_requests.assert_not_awaited()

    async def test_done_event_set_during_parse_skips_enrichment(
        self, extractor, progress, handle_book, monkeypatch
    ):
        """The second done_event check (after parsing) should also short-circuit, even for a
        successfully parsed book -- covers a concurrent handler completing the run mid-parse.
        """

        def _parse_and_complete(soup, book_id):
            progress.done_event.set()
            return {"is_nonexistent": False, "editions_url": "http://editions", "first_author_url": "http://author"}

        monkeypatch.setattr(extractor, "_parse_book_page", _parse_and_complete)

        context = _context(Request.from_url("http://x", label="book", user_data={"book_id": "123"}))
        await handle_book(context)

        assert progress.pending == {}
        context.add_requests.assert_not_awaited()

    async def test_valid_book_with_both_urls_enqueues_both_and_stays_pending(
        self, extractor, progress, handle_book, monkeypatch
    ):
        """A book with both an author URL and an editions URL should enqueue both requests and
        remain in progress.pending, not yet finalized.
        """
        monkeypatch.setattr(
            extractor,
            "_parse_book_page",
            lambda soup, book_id: {
                "is_nonexistent": False,
                "editions_url": "http://editions",
                "first_author_url": "http://author",
            },
        )

        context = _context(Request.from_url("http://x", label="book", user_data={"book_id": "123"}))
        await handle_book(context)

        assert "123" in progress.pending
        assert progress.pending["123"]["_awaiting_author"] is True
        assert progress.pending["123"]["_awaiting_editions"] is True
        assert progress.pending_editions_lookup["http://editions"] == "123"
        assert context.add_requests.await_count == 2
        assert progress.successes == 0

    async def test_valid_book_with_no_urls_finalizes_immediately(self, extractor, handle_book, monkeypatch):
        """A book with neither enrichment URL should be finalized in the same call, with no
        replacement enqueued when target_count is exactly met by this one record.
        """
        progress = BookExtractor._Progress(target_count=1, tried_book_ids=set())
        crawler = _FakeCrawler()
        extractor._attach_handlers(crawler, progress)
        handle_book = crawler.router.handlers[extract_module.LABEL_BOOK]

        monkeypatch.setattr(
            extractor,
            "_parse_book_page",
            lambda soup, book_id: {"is_nonexistent": False, "editions_url": None, "first_author_url": None},
        )

        context = _context(Request.from_url("http://x", label="book", user_data={"book_id": "123"}))
        await handle_book(context)

        assert progress.successes == 1
        assert progress.is_complete is True
        context.add_requests.assert_not_awaited()

    async def test_missing_book_id_raises(self, handle_book):
        """handle_book calls _require_book_id, so a request with no valid book_id should raise."""
        context = _context(Request.from_url("http://x", label="book", user_data={}))
        with pytest.raises(ValueError, match="Missing or invalid book_id"):
            await handle_book(context)


class TestHandleAuthor:
    """Tests for the handle_author route handler wired up by _attach_handlers."""

    @pytest.fixture
    def progress(self):
        return BookExtractor._Progress(target_count=3, tried_book_ids=set())

    @pytest.fixture
    def handle_author(self, extractor, progress):
        crawler = _FakeCrawler()
        extractor._attach_handlers(crawler, progress)
        return crawler.router.handlers[extract_module.LABEL_AUTHOR]

    async def test_merges_author_stats_into_first_contributor(self, extractor, progress, handle_author, monkeypatch):
        """Author stats should be merged into contributors[0], preserving its existing fields."""
        progress.pending["123"] = {
            "contributors": [{"name": "A", "role": "Author"}],
            "_awaiting_author": True,
            "_awaiting_editions": False,
        }
        monkeypatch.setattr(extractor, "_parse_author_page", lambda soup: {"num_followers": 10, "num_distinct_works": 5})

        context = _context(Request.from_url("http://x", label="author", user_data={"book_id": "123"}))
        await handle_author(context)

        assert progress.records[0]["contributors"][0] == {
            "name": "A",
            "role": "Author",
            "num_followers": 10,
            "num_distinct_works": 5,
        }
        assert "_awaiting_author" not in progress.records[0]

    async def test_empty_contributors_still_clears_awaiting_flag(self, extractor, progress, handle_author, monkeypatch):
        """If contributors is empty, the merge is skipped, but _awaiting_author is still cleared."""
        progress.pending["123"] = {"contributors": [], "_awaiting_author": True, "_awaiting_editions": False}
        monkeypatch.setattr(extractor, "_parse_author_page", lambda soup: {"num_followers": 1, "num_distinct_works": 1})

        context = _context(Request.from_url("http://x", label="author", user_data={"book_id": "123"}))
        await handle_author(context)

        assert progress.records[0]["contributors"] == []
        assert progress.successes == 1

    async def test_missing_pending_record_is_noop(self, extractor, progress, handle_author, monkeypatch):
        """An author callback for a book_id with no pending record should do nothing."""
        monkeypatch.setattr(extractor, "_parse_author_page", lambda soup: {"num_followers": 1, "num_distinct_works": 1})

        context = _context(Request.from_url("http://x", label="author", user_data={"book_id": "unknown"}))
        await handle_author(context)

        assert progress.pending == {}
        assert progress.records == []

    async def test_missing_book_id_raises(self, handle_author):
        """handle_author calls _require_book_id, so a request with no valid book_id should raise."""
        context = _context(Request.from_url("http://x", label="author", user_data={"book_id": 123}))
        with pytest.raises(ValueError, match="Missing or invalid book_id"):
            await handle_author(context)


class TestHandleEditions:
    """Tests for the handle_editions route handler wired up by _attach_handlers."""

    @pytest.fixture
    def progress(self):
        return BookExtractor._Progress(target_count=3, tried_book_ids=set())

    @pytest.fixture
    def handle_editions(self, extractor, progress):
        crawler = _FakeCrawler()
        extractor._attach_handlers(crawler, progress)
        return crawler.router.handlers[extract_module.LABEL_EDITIONS]

    @pytest.mark.parametrize(
        "num_editions, expected_has_more",
        [(5, True), (1, False), (None, False)],
        ids=["multiple_editions", "single_edition", "parse_failure"],
    )
    async def test_sets_num_editions_and_has_more_editions(
        self, extractor, progress, handle_editions, monkeypatch, num_editions, expected_has_more
    ):
        """num_editions and has_more_editions should be set from _parse_editions_page's result.

        Covers more than one edition (has_more=True), exactly one
        edition (has_more=False), and a parse failure (None, has_more=
        False -- bool(None and ...) must short-circuit safely, not raise).
        """
        progress.pending["123"] = {"contributors": [], "_awaiting_author": False, "_awaiting_editions": True}
        monkeypatch.setattr(extractor, "_parse_editions_page", lambda soup: num_editions)

        context = _context(Request.from_url("http://editions", label="editions", user_data={"book_id": "123"}))
        await handle_editions(context)

        assert progress.records[0]["num_editions"] == num_editions
        assert progress.records[0]["has_more_editions"] is expected_has_more

    async def test_pops_pending_editions_lookup_unconditionally(self, extractor, progress, handle_editions):
        """pending_editions_lookup should be cleared even when no matching pending record exists
        (e.g. a stray callback for a book that already finalized through another path).
        """
        progress.pending_editions_lookup["http://editions"] = "999"

        context = _context(Request.from_url("http://editions", label="editions", user_data={"book_id": "999"}))
        await handle_editions(context)

        assert "http://editions" not in progress.pending_editions_lookup
        assert progress.records == []

    async def test_missing_book_id_raises(self, handle_editions):
        """handle_editions calls _require_book_id, so a request with no valid book_id should raise."""
        context = _context(Request.from_url("http://editions", label="editions", user_data={}))
        with pytest.raises(ValueError, match="Missing or invalid book_id"):
            await handle_editions(context)


class TestHandleFailed:
    """Tests for the failed_request_handler wired up by _attach_handlers.

    Unlike the three page handlers, this one uses _optional_book_id (not
    _require_book_id), so it degrades to a no-op on a missing/invalid
    book_id instead of raising.
    """

    @pytest.fixture
    def progress(self):
        return BookExtractor._Progress(target_count=3, tried_book_ids=set())

    @pytest.fixture
    def handle_failed(self, extractor, progress):
        crawler = _FakeCrawler()
        extractor._attach_handlers(crawler, progress)
        return crawler.failed_handler

    @pytest.mark.parametrize(
        "label, user_data",
        [("author", {"book_id": 123}), ("author", {"book_id": "no-such-record"})],
        ids=["unresolvable_book_id", "no_pending_record"],
    )
    async def test_noop_cases(self, progress, handle_failed, label, user_data):
        """handle_failed should be a pure no-op when book_id is unresolvable (returns at
        `if book_id is None`), or when book_id is valid but nothing is pending under it
        (returns at `if record is None`).
        """
        context = _context(Request.from_url("http://x", label=label, user_data=user_data))
        result = await handle_failed(context, Exception("boom"))

        assert result is None
        assert progress.pending == {}
        assert progress.records == []

    async def test_unknown_label_skips_flag_clearing_but_still_finalizes(self, progress, handle_failed):
        """A label matching neither LABEL_AUTHOR nor LABEL_EDITIONS should skip both
        flag-clearing branches and fall through directly to _try_finalize.
        """
        progress.pending["123"] = {"contributors": [], "_awaiting_author": False, "_awaiting_editions": False}

        context = _context(Request.from_url("http://x", label="some_other_label", user_data={"book_id": "123"}))
        result = await handle_failed(context, Exception("boom"))

        assert result is None
        assert progress.successes == 1

    @pytest.mark.parametrize(
        "book_id, user_data",
        [("123", {"book_id": "123"}), (None, {})],
        ids=["with_book_id", "without_book_id"],
    )
    async def test_book_failure_enqueues_replacement(self, progress, handle_failed, monkeypatch, book_id, user_data):
        """A permanently failed book request should always enqueue a replacement, whether or
        not book_id resolved -- covers both sides of `if book_id is not None:`.
        """
        if book_id:
            progress.pending[book_id] = {"contributors": [], "_awaiting_author": True, "_awaiting_editions": True}
        monkeypatch.setattr(BookExtractor._Progress, "next_random_book_id", lambda self: "999")

        context = _context(Request.from_url("http://x", label=extract_module.LABEL_BOOK, user_data=user_data))
        result = await handle_failed(context, Exception("boom"))

        assert result is None
        if book_id:
            assert book_id not in progress.pending
        context.add_requests.assert_awaited_once()

    @pytest.mark.parametrize("label", ["author", "editions"], ids=["author_failure", "editions_failure"])
    async def test_author_or_editions_failure_clears_flag_and_finalizes(self, progress, handle_failed, label):
        """A permanently failed author or editions request should clear its own _awaiting_*
        flag and finalize, since the other enrichment step may already be done.
        """
        flag = "_awaiting_author" if label == "author" else "_awaiting_editions"
        progress.pending["123"] = {"contributors": [], "_awaiting_author": False, "_awaiting_editions": False}
        progress.pending["123"][flag] = True

        context = _context(Request.from_url("http://x", label=label, user_data={"book_id": "123"}))
        result = await handle_failed(context, Exception("boom"))

        assert result is None
        assert progress.successes == 1
        assert flag not in progress.records[0]

    async def test_editions_failure_clears_pending_editions_lookup(self, progress, handle_failed):
        """An editions failure should additionally clear the URL->book_id lookup entry --
        the one side effect unique to the LABEL_EDITIONS branch.
        """
        progress.pending["123"] = {"contributors": [], "_awaiting_author": False, "_awaiting_editions": True}
        progress.pending_editions_lookup["http://x"] = "123"

        context = _context(Request.from_url("http://x", label=extract_module.LABEL_EDITIONS, user_data={"book_id": "123"}))
        await handle_failed(context, Exception("boom"))

        assert "http://x" not in progress.pending_editions_lookup


class TestDriveToCompletion:
    """Tests for _drive_to_completion."""

    async def test_returns_when_crawler_run_finishes_naturally(self, extractor):
        """If crawler.run() completes on its own, _drive_to_completion should simply return."""
        progress = BookExtractor._Progress(target_count=1, tried_book_ids=set())
        crawler = SimpleNamespace(run=AsyncMock(return_value=None))

        await extractor._drive_to_completion(crawler, [], progress)

        assert not progress.done_event.is_set()

    async def test_cancels_run_task_when_done_event_fires_first(self, extractor):
        """If done_event is set before crawler.run() finishes, the run task should be cancelled
        rather than awaited to completion.
        """
        progress = BookExtractor._Progress(target_count=1, tried_book_ids=set())
        run_started = asyncio.Event()

        async def _fake_run(requests):
            run_started.set()
            await asyncio.sleep(10)  # would hang the test if not cancelled

        crawler = SimpleNamespace(run=_fake_run)

        async def _set_done_soon():
            await run_started.wait()
            progress.done_event.set()

        setter_task = asyncio.create_task(_set_done_soon())
        await asyncio.wait_for(extractor._drive_to_completion(crawler, [], progress), timeout=5)
        await setter_task

        assert progress.done_event.is_set()


class TestResetRequestQueueStorage:
    """Tests for _reset_request_queue_storage."""

    async def test_opens_and_drops_the_default_queue(self, extractor, monkeypatch):
        """_reset_request_queue_storage should open the default RequestQueue and drop it."""
        fake_queue = AsyncMock()
        open_mock = AsyncMock(return_value=fake_queue)
        monkeypatch.setattr(extract_module.RequestQueue, "open", open_mock)

        await extractor._reset_request_queue_storage()

        open_mock.assert_awaited_once()
        fake_queue.drop.assert_awaited_once()


# ---------------------------------------------------------------------------
# GROUP 6 -- extract() orchestration
# ---------------------------------------------------------------------------


class TestExtractOrchestration:
    """Tests for the public extract() entry point's orchestration logic.

    The real Crawlee crawl is never run here -- _build_crawler,
    _attach_handlers, and _drive_to_completion are all mocked so these
    tests verify extract()'s own control flow (seeding, early returns)
    without touching the network.
    """

    @pytest.fixture(autouse=True)
    def _mock_request_queue(self, monkeypatch):
        """Patch RequestQueue.open/drop so _reset_request_queue_storage needs no real storage backend."""
        fake_queue = AsyncMock()
        monkeypatch.setattr(extract_module.RequestQueue, "open", AsyncMock(return_value=fake_queue))

    async def test_returns_empty_list_for_non_positive_target(self, extractor):
        """target_count <= 0 should short-circuit to an empty list with no crawl attempted."""
        assert await extractor.extract(0, set()) == []
        assert await extractor.extract(-5, set()) == []

    async def test_returns_empty_list_when_no_ids_available(self, extractor, monkeypatch):
        """If no untried book IDs can be drawn at all, extract() should return [] with a warning."""
        monkeypatch.setattr(
            BookExtractor._Progress, "next_random_book_id", lambda self: None
        )
        result = await extractor.extract(5, set())
        assert result == []

    async def test_seeds_crawl_and_returns_progress_records(self, extractor, monkeypatch):
        """extract() should build a crawler, attach handlers, drive to completion, and return records."""
        fake_records = [{"book_id": "1"}, {"book_id": "2"}]

        async def _fake_drive_to_completion(self, crawler, seed_requests, progress):
            progress.records.extend(fake_records)

        monkeypatch.setattr(extract_module.BookExtractor, "_build_crawler", lambda self: MagicMock())
        monkeypatch.setattr(extract_module.BookExtractor, "_attach_handlers", lambda self, crawler, progress: None)
        monkeypatch.setattr(extract_module.BookExtractor, "_drive_to_completion", _fake_drive_to_completion)

        result = await extractor.extract(2, set())
        assert result == fake_records

    async def test_truncates_records_to_target_count(self, extractor, monkeypatch):
        """extract() should never return more than target_count records, even if more were collected."""

        async def _fake_drive_to_completion(self, crawler, seed_requests, progress):
            progress.records.extend([{"book_id": str(i)} for i in range(5)])

        monkeypatch.setattr(extract_module.BookExtractor, "_build_crawler", lambda self: MagicMock())
        monkeypatch.setattr(extract_module.BookExtractor, "_attach_handlers", lambda self, crawler, progress: None)
        monkeypatch.setattr(extract_module.BookExtractor, "_drive_to_completion", _fake_drive_to_completion)

        result = await extractor.extract(2, set())
        assert len(result) == 2

    async def test_logs_warning_when_crawl_ends_early(self, extractor, monkeypatch):
        """If the drive loop ends with fewer records than target_count, extract() should log a warning."""

        async def _fake_drive_to_completion(self, crawler, seed_requests, progress):
            progress.records.extend([{"book_id": "1"}])  # only 1 of 3 requested

        monkeypatch.setattr(extract_module.BookExtractor, "_build_crawler", lambda self: MagicMock())
        monkeypatch.setattr(extract_module.BookExtractor, "_attach_handlers", lambda self, crawler, progress: None)
        monkeypatch.setattr(extract_module.BookExtractor, "_drive_to_completion", _fake_drive_to_completion)

        messages = []
        sink_id = logger.add(lambda msg: messages.append(msg), level="WARNING")
        try:
            result = await extractor.extract(3, set())
        finally:
            logger.remove(sink_id)

        assert result == [{"book_id": "1"}]
        assert any("Crawl ended early" in m for m in messages)
