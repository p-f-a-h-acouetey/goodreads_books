"""Pytest suite for src.goodreads_etl.extractors.extract."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from bs4 import BeautifulSoup

from src.goodreads_etl.extractors.extract import (
    _attach_handlers,
    _build_author_request,
    _build_book_request,
    _build_crawlee_crawler,
    _build_record_from_book_page,
    _drive_crawl_to_completion,
    _require_book_id,
    _reset_request_queue_storage,
    _SamplingState,
    _seed_initial_requests,
    run_sampling_crawl,
)
from src.goodreads_etl.utils.book_recorder import BookRecord
from src.goodreads_etl.utils.config_setter import Settings

# ---------------------------------------------------------------------------
# Shared fixtures / factories
# ---------------------------------------------------------------------------


def make_settings(**overrides: Any) -> Settings:
    """Factory helper to instantiate a `Settings` instance with default test parameters.

    Args:
        **overrides: Optional key-value pairs to override default setting values.

    Returns:
        Settings: A configured Settings object for testing.
    """
    defaults = dict(
        max_concurrency=2,
        max_tasks_per_minute=60,
        num_retries=3,
        timeout_seconds=30,
        use_session_pool=False,
        retry_on_blocked=True,
        max_session_pool_size=10,
        additional_http_error_status_codes=(),
        max_session_rotations=3,
        max_random_id_attempts=1000,
        label_book="BOOK",
        label_author="AUTHOR",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_state(**overrides: Any) -> _SamplingState:
    """Factory helper to instantiate a `_SamplingState` tracker.

    Args:
        **overrides: Optional key-value pairs to override state configuration attributes.

    Returns:
        _SamplingState: A initialized sampling state object for tracking crawl progress.
    """
    defaults = dict(sample_size=3, min_book_id=1, max_book_id=100, max_random_id_attempts=1000)
    defaults.update(overrides)
    return _SamplingState(**defaults)


def make_record(**overrides: Any) -> BookRecord:
    """Factory helper to construct a dummy `BookRecord` instance.

    Args:
        **overrides: Optional field overrides for the default book attributes.

    Returns:
        BookRecord: A sample BookRecord populated with test data.
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


def make_soup(html_str: str) -> BeautifulSoup:
    """Parse raw HTML string into a BeautifulSoup context object using lxml.

    Args:
        html_str: The HTML markup string to parse.

    Returns:
        BeautifulSoup: Parsed HTML DOM document object.
    """
    return BeautifulSoup(html_str, "lxml")


def build_book_context(*, book_id: str, html_str: str) -> MagicMock:
    """Construct a mock Crawlee context object simulating a book page request.

    Args:
        book_id: Unique string identifier for the target book.
        html_str: Raw HTML payload string associated with the simulated page response.

    Returns:
        MagicMock: A Crawlee context mock configured with book user metadata and DOM soup.
    """
    context = MagicMock()
    context.request.user_data = {"book_id": book_id}
    context.soup = make_soup(html_str)
    context.add_requests = AsyncMock()
    return context


def attach_and_capture_raw_failed_handler(*, crawler: Any, state: Any, settings: Any) -> Any:
    """Capture the raw (unwrapped) failed_request_handler function.

    Crawlee's `failed_request_handler` decorator wraps the function in
    `_wrap_handler_with_error_context`, which calls `context.create_modified_copy()`
    internally. On a MagicMock context that returns a disconnected mock, so
    `context.request.label` never matches inside the handler. Capturing the
    raw function before wrapping sidesteps that entirely.

    Args:
        crawler: Target Crawlee crawler instance.
        state: Internal `_SamplingState` instance to bind.
        settings: Application `Settings` instance.

    Returns:
        Callable: The original unwrapped error-handling coroutine function.
    """
    captured = {}
    original_decorator = crawler.failed_request_handler

    def capturing_decorator(handler: Any) -> Any:
        captured["handler"] = handler
        return original_decorator(handler)

    crawler.failed_request_handler = capturing_decorator
    _attach_handlers(crawler=crawler, state=state, settings=settings)
    return captured["handler"]


def build_failed_context(*, label: str, book_id: str) -> MagicMock:
    """Construct a mock Crawlee context object simulating a failed request scenario.

    Args:
        label: Request label string identifying route target (e.g. 'BOOK' or 'AUTHOR').
        book_id: Unique book ID bound to the request user metadata.

    Returns:
        MagicMock: A Crawlee context mock configured to represent a failed request payload.
    """
    context = MagicMock()
    context.request.label = label
    context.request.user_data = {"book_id": book_id}
    context.add_requests = AsyncMock()
    return context


BOOK_JSON_LD_HTML = '<script type="application/ld+json">{"@type": "Book", "name": "Dune"}</script>'
BOOK_JSON_LD_WITH_AUTHOR_HTML = BOOK_JSON_LD_HTML + '<a href="/author/show/1">Frank Herbert</a>'
NO_JSON_LD_HTML = "<html><p>no ld+json here</p></html>"


# ---------------------------------------------------------------------------
# _SamplingState
# ---------------------------------------------------------------------------


class TestSamplingStateSuccessesAndCompletion:
    """Tests evaluating target state counting and completion status logic."""

    def test_successes_reflects_record_count(self) -> None:
        """Verify `successes` property correctly reflects total collected record count."""
        state = make_state(sample_size=5)
        state.records.append(make_record())
        assert state.successes == 1

    def test_is_complete_false_when_below_target(self) -> None:
        """Verify `is_complete` remains False while success count is below target `sample_size`."""
        state = make_state(sample_size=3)
        state.records.append(make_record())
        assert state.is_complete is False

    def test_is_complete_true_when_target_met_or_exceeded(self) -> None:
        """Verify `is_complete` returns True when collected records meet or exceed target size."""
        state = make_state(sample_size=1)
        state.records.append(make_record(book_id="1"))
        state.records.append(make_record(book_id="2"))
        assert state.is_complete is True


class TestSamplingStateNextRandomBookId:
    """Tests evaluating random candidate book ID generation and exhaustion limits."""

    def test_returns_id_within_range_and_marks_as_tried(self) -> None:
        """Verify generated random book ID falls within bounds and gets marked in `tried_book_ids`."""
        state = make_state(min_book_id=1, max_book_id=10)
        book_id = state.next_random_book_id()
        assert 1 <= int(book_id) <= 10
        assert book_id in state.tried_book_ids

    def test_raises_runtime_error_when_range_exhausted(self) -> None:
        """Verify `RuntimeError` is raised when candidate ID space attempts exceed max threshold."""
        state = make_state(min_book_id=1, max_book_id=1, max_random_id_attempts=5)
        state.next_random_book_id()
        with pytest.raises(RuntimeError):
            state.next_random_book_id()


class TestSamplingStateAddRecord:
    """Tests evaluating record append operations and completion event triggers."""

    def test_adds_record_when_capacity_remains(self) -> None:
        """Verify record addition succeeds and updates state when quota is not yet met."""
        state = make_state(sample_size=2)
        assert state.add_record(record=make_record()) is True
        assert state.successes == 1

    def test_returns_false_and_stops_growth_when_already_complete(self) -> None:
        """Verify record addition fails and rejects additions after sample target is satisfied."""
        state = make_state(sample_size=1)
        state.add_record(record=make_record(book_id="1"))
        assert state.add_record(record=make_record(book_id="2")) is False
        assert state.successes == 1

    def test_sets_done_event_only_when_target_reached(self) -> None:
        """Verify `done_event` asyncio signal is set if and only if sample target is satisfied."""
        state = make_state(sample_size=3)
        state.add_record(record=make_record())
        assert state.done_event.is_set() is False

        state.add_record(record=make_record(book_id="2"))
        state.add_record(record=make_record(book_id="3"))
        assert state.done_event.is_set() is True


# ---------------------------------------------------------------------------
# _require_book_id
# ---------------------------------------------------------------------------


class TestRequireBookId:
    """Tests evaluating strict book_id extraction from request user_data."""

    def test_returns_book_id_when_present_and_valid(self) -> None:
        """Verify a valid string book_id present in user_data is returned unchanged."""
        context = build_book_context(book_id="1", html_str="<html></html>")
        assert _require_book_id(request=context.request) == "1"

    def test_raises_value_error_when_book_id_missing(self) -> None:
        """Verify ValueError is raised when user_data lacks a valid book_id."""
        context = build_book_context(book_id="1", html_str="<html></html>")
        context.request.user_data = {}

        with pytest.raises(ValueError, match="Missing or invalid book_id"):
            _require_book_id(request=context.request)


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


class TestBuildBookRequest:
    """Tests evaluating proper construction of book Crawlee Request objects."""

    def test_builds_request_with_correct_url_label_and_keys(self) -> None:
        """Verify book request is assembled with correct target URL, route label, and metadata keys."""
        settings = make_settings()
        request = _build_book_request(book_id="123", settings=settings)
        assert request.url == f"{settings.base_book_url}123"
        assert request.label == "BOOK"
        assert request.user_data["book_id"] == "123"
        assert request.unique_key == "book-123"


class TestBuildAuthorRequest:
    """Tests evaluating proper construction of author Crawlee Request objects."""

    def test_builds_request_with_correct_url_label_and_keys(self) -> None:
        """Verify author request is assembled with target author URL, route label, and correlation metadata."""
        settings = make_settings()
        request = _build_author_request(
            author_url="https://www.goodreads.com/author/show/9",
            book_id="123",
            settings=settings,
        )
        assert request.url == "https://www.goodreads.com/author/show/9"
        assert request.label == "AUTHOR"
        assert request.user_data["book_id"] == "123"
        assert request.unique_key == "author-123"


# ---------------------------------------------------------------------------
# _build_record_from_book_page
# ---------------------------------------------------------------------------


class TestBuildRecordFromBookPage:
    """Tests evaluating record extraction and normalization from parsed HTML book pages."""

    def test_assembles_record_with_full_json_ld(self) -> None:
        """Verify `BookRecord` is constructed accurately when valid JSON-LD and page DOM are present."""
        settings = make_settings()
        soup = make_soup(
            '<a href="/author/show/1">Frank Herbert</a><p>412 pages</p><p>1,000 reviews</p>'
        )
        json_ld = {
            "name": "Dune",
            "aggregateRating": {"ratingValue": "4.5", "reviewCount": 5000},
            "inLanguage": "en",
        }
        record, first_author_url = _build_record_from_book_page(
            book_id="1", soup=soup, json_ld=json_ld, next_data=None, settings=settings
        )
        assert record.book_id == "1"
        assert record.title == "dune"
        assert record.first_author == "frank herbert"
        assert first_author_url == "https://www.goodreads.com/author/show/1"
        assert record.average_rating == 4.5
        assert record.language_code == "en"
        assert record.url == f"{settings.base_book_url}1"

    @pytest.mark.parametrize(
        "json_ld",
        [
            {"name": "Dune"},
            {"name": "Dune", "aggregateRating": None},
        ],
    )
    def test_missing_or_null_aggregate_rating_defaults_to_zero(
        self, json_ld: dict[str, Any]
    ) -> None:
        """Verify average rating gracefully falls back to 0.0 when missing or explicitly null in JSON-LD."""
        settings = make_settings()
        record, _ = _build_record_from_book_page(
            book_id="1",
            soup=make_soup("<html></html>"),
            json_ld=json_ld,
            next_data=None,
            settings=settings,
        )
        assert record.average_rating == 0.0

    def test_missing_fields_default_to_empty_string_and_no_author_url(self) -> None:
        """Verify absent metadata defaults gracefully to empty strings, zero values, and None for author URL."""
        settings = make_settings()
        record, first_author_url = _build_record_from_book_page(
            book_id="1",
            soup=make_soup("<html></html>"),
            json_ld={},
            next_data=None,
            settings=settings,
        )
        assert record.title == ""
        assert record.first_author == ""
        assert record.first_author_url == ""
        assert first_author_url is None
        assert record.num_pages == 0
        assert record.num_reviews == 0

    def test_next_data_passed_through_to_publisher_extraction(self) -> None:
        """Verify `next_data` JSON payload is passed downstream and used for publisher field extraction."""
        settings = make_settings()
        record, _ = _build_record_from_book_page(
            book_id="1",
            soup=make_soup("<html></html>"),
            json_ld={},
            next_data={"publisher": "Ace Books"},
            settings=settings,
        )
        assert record.publisher == "ace books"


# ---------------------------------------------------------------------------
# _build_crawlee_crawler
# ---------------------------------------------------------------------------


class TestBuildCrawleeCrawler:
    """Tests evaluating instantiation and configuration of Crawlee crawler objects."""

    @pytest.mark.parametrize("max_concurrency", [0, None, 5])
    def test_builds_crawler_regardless_of_concurrency_setting(
        self, max_concurrency: int | None
    ) -> None:
        """Verify crawler is instantiated without error across various concurrency config parameters."""
        assert (
            _build_crawlee_crawler(settings=make_settings(max_concurrency=max_concurrency))
            is not None
        )

    @pytest.mark.parametrize("use_session_pool", [True, False])
    def test_builds_crawler_regardless_of_session_pool_setting(
        self, use_session_pool: bool
    ) -> None:
        """Verify crawler is built successfully regardless of whether session pooling is enabled or disabled."""
        settings = make_settings(use_session_pool=use_session_pool, max_session_pool_size=50)
        assert _build_crawlee_crawler(settings=settings) is not None


# ---------------------------------------------------------------------------
# _seed_initial_requests
# ---------------------------------------------------------------------------


class TestSeedInitialRequests:
    """Tests evaluating generation of initial batch requests for crawler startup."""

    def test_returns_seed_requests_up_to_double_concurrency(self) -> None:
        """Verify seed request batch size scales up to twice the configured concurrency limit."""
        settings = make_settings(max_concurrency=2)
        state = make_state(sample_size=10, min_book_id=1, max_book_id=1000)
        requests = _seed_initial_requests(state=state, settings=settings)
        assert len(requests) == 8
        assert len(state.tried_book_ids) == 8

    def test_seed_count_capped_by_sample_size(self) -> None:
        """Verify initial seed request count never exceeds the target total `sample_size`."""
        settings = make_settings(max_concurrency=5)
        state = make_state(sample_size=2, min_book_id=1, max_book_id=1000)
        assert len(_seed_initial_requests(state=state, settings=settings)) == 2

    def test_seed_count_at_least_one_when_concurrency_zero(self) -> None:
        """Verify at least two seed requests are enqueued even if concurrency setting is zero."""
        settings = make_settings(max_concurrency=0)
        state = make_state(sample_size=5, min_book_id=1, max_book_id=1000)
        assert len(_seed_initial_requests(state=state, settings=settings)) == 4

    def test_returns_none_when_id_space_exhausted(self) -> None:
        """Verify returns None when candidate ID space is completely exhausted during initial seeding."""
        settings = make_settings(max_concurrency=1)
        state = make_state(sample_size=5, min_book_id=1, max_book_id=1, max_random_id_attempts=5)
        assert _seed_initial_requests(state=state, settings=settings) is None


# ---------------------------------------------------------------------------
# handle_book (via router._handlers_by_label)
# ---------------------------------------------------------------------------


class TestHandleBook:
    """Tests evaluating execution logic for the book page router handler."""

    @pytest.mark.asyncio
    async def test_skips_when_done_event_already_set(self) -> None:
        """Verify book processing exits immediately with no request additions if state is already completed."""
        settings = make_settings()
        state = make_state(sample_size=1)
        state.done_event.set()
        crawler = _build_crawlee_crawler(settings=settings)
        _attach_handlers(crawler=crawler, state=state, settings=settings)

        context = build_book_context(book_id="1", html_str="<html></html>")
        handler = crawler.router._handlers_by_label[settings.label_book]
        await handler(context)

        context.add_requests.assert_not_called()

    @pytest.mark.asyncio
    async def test_enqueues_replacement_when_no_json_ld(self) -> None:
        """Verify missing JSON-LD triggers a replacement book request enqueue."""
        settings = make_settings()
        state = make_state(sample_size=3, min_book_id=1, max_book_id=1000)
        crawler = _build_crawlee_crawler(settings=settings)
        _attach_handlers(crawler=crawler, state=state, settings=settings)

        context = build_book_context(book_id="1", html_str=NO_JSON_LD_HTML)
        handler = crawler.router._handlers_by_label[settings.label_book]
        await handler(context)

        context.add_requests.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_json_ld_skips_enqueue_when_already_complete(self) -> None:
        """Verify replacement book is not enqueued for missing JSON-LD if state target reached concurrently."""
        settings = make_settings()
        state = make_state(sample_size=1, min_book_id=1, max_book_id=1000)
        state.records.append(make_record(book_id="already-done"))
        assert state.is_complete is True

        crawler = _build_crawlee_crawler(settings=settings)
        _attach_handlers(crawler=crawler, state=state, settings=settings)

        context = build_book_context(book_id="2", html_str=NO_JSON_LD_HTML)
        handler = crawler.router._handlers_by_label[settings.label_book]
        await handler(context)

        context.add_requests.assert_not_called()
        assert len(state.tried_book_ids) == 0

    @pytest.mark.asyncio
    async def test_no_json_ld_logs_warning_when_id_space_exhausted(self, mocker: Any) -> None:
        """Verify warning is logged when missing JSON-LD cannot enqueue a replacement due to ID exhaustion."""
        settings = make_settings()
        state = make_state(sample_size=5, min_book_id=1, max_book_id=1, max_random_id_attempts=5)
        state.tried_book_ids.add("1")

        crawler = _build_crawlee_crawler(settings=settings)
        _attach_handlers(crawler=crawler, state=state, settings=settings)

        warning_mock = mocker.patch("src.goodreads_etl.extractors.extract.logger.warning")
        context = build_book_context(book_id="1", html_str=NO_JSON_LD_HTML)
        handler = crawler.router._handlers_by_label[settings.label_book]
        await handler(context)

        context.add_requests.assert_not_called()
        assert any(
            "No replacement book ID available" in str(call.args[0])
            for call in warning_mock.call_args_list
        )

    @pytest.mark.asyncio
    async def test_enqueues_author_request_when_author_url_found(self) -> None:
        """Verify finding an author URL stores pending record and enqueues an author detail request."""
        settings = make_settings()
        state = make_state(sample_size=3, min_book_id=1, max_book_id=1000)
        crawler = _build_crawlee_crawler(settings=settings)
        _attach_handlers(crawler=crawler, state=state, settings=settings)

        context = build_book_context(book_id="1", html_str=BOOK_JSON_LD_WITH_AUTHOR_HTML)
        handler = crawler.router._handlers_by_label[settings.label_book]
        await handler(context)

        assert "1" in state.pending_author_records
        context.add_requests.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_finalizes_directly_when_no_author_url(self) -> None:
        """Verify record is finalized immediately to state when page contains no secondary author URL."""
        settings = make_settings()
        state = make_state(sample_size=3, min_book_id=1, max_book_id=1000)
        crawler = _build_crawlee_crawler(settings=settings)
        _attach_handlers(crawler=crawler, state=state, settings=settings)

        context = build_book_context(book_id="1", html_str=BOOK_JSON_LD_HTML)
        handler = crawler.router._handlers_by_label[settings.label_book]
        await handler(context)

        assert state.successes == 1

    @pytest.mark.asyncio
    async def test_finalize_no_ops_when_state_already_complete(self) -> None:
        """Verify record finalization is safely ignored if sample target was satisfied mid-flight."""
        settings = make_settings()
        state = make_state(sample_size=1, min_book_id=1, max_book_id=1000)
        state.records.append(make_record(book_id="already-done"))
        assert state.is_complete is True

        crawler = _build_crawlee_crawler(settings=settings)
        _attach_handlers(crawler=crawler, state=state, settings=settings)

        context = build_book_context(book_id="2", html_str=BOOK_JSON_LD_HTML)
        handler = crawler.router._handlers_by_label[settings.label_book]
        await handler(context)

        context.add_requests.assert_not_called()
        assert state.successes == 1

    @pytest.mark.asyncio
    async def test_skips_finalization_when_done_event_set_after_parsing(self, mocker: Any) -> None:
        """Verify finalization is aborted if completion event fires during JSON-LD parsing step."""
        settings = make_settings()
        state = make_state(sample_size=1, min_book_id=1, max_book_id=1000)
        crawler = _build_crawlee_crawler(settings=settings)
        _attach_handlers(crawler=crawler, state=state, settings=settings)

        import src.goodreads_etl.extractors.extract as extract_mod

        original_extract = extract_mod.extract_json_ld

        def side_effect(*args: Any, **kwargs: Any) -> Any:
            state.done_event.set()
            return original_extract(*args, **kwargs)

        mocker.patch.object(extract_mod, "extract_json_ld", side_effect=side_effect)

        context = build_book_context(book_id="1", html_str=BOOK_JSON_LD_WITH_AUTHOR_HTML)
        handler = crawler.router._handlers_by_label[settings.label_book]
        await handler(context)

        assert "1" not in state.pending_author_records
        context.add_requests.assert_not_called()


# ---------------------------------------------------------------------------
# handle_author (via router._handlers_by_label)
# ---------------------------------------------------------------------------


class TestHandleAuthor:
    """Tests evaluating execution logic for the author page router handler."""

    @pytest.mark.asyncio
    async def test_returns_when_no_pending_record(self) -> None:
        """Verify handler exits without side effects when no pending book record matches the request."""
        settings = make_settings()
        state = make_state(sample_size=3)
        crawler = _build_crawlee_crawler(settings=settings)
        _attach_handlers(crawler=crawler, state=state, settings=settings)

        context = build_book_context(book_id="unknown", html_str="<html></html>")
        handler = crawler.router._handlers_by_label[settings.label_author]
        await handler(context)

        assert state.successes == 0

    @pytest.mark.asyncio
    async def test_merges_stats_and_finalizes_record(self) -> None:
        """Verify author statistics are parsed from page and merged into pending record before finalization."""
        settings = make_settings()
        state = make_state(sample_size=1)
        record = make_record(book_id="1")
        state.pending_author_records["1"] = record
        crawler = _build_crawlee_crawler(settings=settings)
        _attach_handlers(crawler=crawler, state=state, settings=settings)

        context = build_book_context(
            book_id="1",
            html_str="<html><body><p>5 distinct works, followers (235,786)</p></body></html>",
        )
        handler = crawler.router._handlers_by_label[settings.label_author]
        await handler(context)

        assert record.first_author_num_books == 5
        assert record.first_author_num_followers == 235_786
        assert state.successes == 1

    @pytest.mark.asyncio
    async def test_defaults_missing_stats(self) -> None:
        """Verify missing author stats default to 1 book and 0 followers upon finalization."""
        settings = make_settings()
        state = make_state(sample_size=1)
        record = make_record(book_id="1")
        state.pending_author_records["1"] = record
        crawler = _build_crawlee_crawler(settings=settings)
        _attach_handlers(crawler=crawler, state=state, settings=settings)

        context = build_book_context(
            book_id="1", html_str="<html><body><p>nothing relevant</p></body></html>"
        )
        handler = crawler.router._handlers_by_label[settings.label_author]
        await handler(context)

        assert record.first_author_num_books == 1
        assert record.first_author_num_followers == 0


# ---------------------------------------------------------------------------
# handle_failed -- captured raw (unwrapped) via attach_and_capture_raw_failed_handler
# ---------------------------------------------------------------------------


class TestHandleFailed:
    """Tests evaluating fallback and cleanup behaviors for failed HTTP requests."""

    @pytest.mark.asyncio
    async def test_failed_book_request_pops_pending_and_enqueues_replacement(self) -> None:
        """Verify failing a book request cleans up pending records and enqueues a new candidate request."""
        settings = make_settings()
        state = make_state(sample_size=3, min_book_id=1, max_book_id=1000)
        state.pending_author_records["1"] = make_record(book_id="1")

        crawler = _build_crawlee_crawler(settings=settings)
        raw_handler = attach_and_capture_raw_failed_handler(
            crawler=crawler, state=state, settings=settings
        )

        context = build_failed_context(label=settings.label_book, book_id="1")
        await raw_handler(context, Exception("boom"))

        assert "1" not in state.pending_author_records
        context.add_requests.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failed_book_request_with_no_book_id_skips_pending_cleanup(self) -> None:
        """Verify failing a book request with no book_id skips pending-record cleanup but still enqueues."""
        settings = make_settings()
        state = make_state(sample_size=3, min_book_id=1, max_book_id=1000)

        crawler = _build_crawlee_crawler(settings=settings)
        raw_handler = attach_and_capture_raw_failed_handler(
            crawler=crawler, state=state, settings=settings
        )

        context = build_failed_context(label=settings.label_book, book_id="1")
        context.request.user_data = {}

        await raw_handler(context, Exception("boom"))

        context.add_requests.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failed_book_request_logs_warning_when_id_space_exhausted(
        self, mocker: Any
    ) -> None:
        """Verify failing a book request logs a warning when no new book IDs remain to enqueue."""
        settings = make_settings()
        state = make_state(sample_size=5, min_book_id=1, max_book_id=1, max_random_id_attempts=5)
        state.tried_book_ids.add("1")

        crawler = _build_crawlee_crawler(settings=settings)
        raw_handler = attach_and_capture_raw_failed_handler(
            crawler=crawler, state=state, settings=settings
        )

        warning_mock = mocker.patch("src.goodreads_etl.extractors.extract.logger.warning")
        context = build_failed_context(label=settings.label_book, book_id="1")
        await raw_handler(context, Exception("boom"))

        context.add_requests.assert_not_called()
        assert any(
            "No replacement book ID available" in str(call.args[0])
            for call in warning_mock.call_args_list
        )

    @pytest.mark.asyncio
    async def test_failed_author_request_finalizes_pending_record(self) -> None:
        """Verify failing an author request still saves the pending book record with default author stats."""
        settings = make_settings()
        state = make_state(sample_size=1)
        state.pending_author_records["1"] = make_record(book_id="1")

        crawler = _build_crawlee_crawler(settings=settings)
        raw_handler = attach_and_capture_raw_failed_handler(
            crawler=crawler, state=state, settings=settings
        )

        context = build_failed_context(label=settings.label_author, book_id="1")
        await raw_handler(context, Exception("boom"))

        assert state.successes == 1

    @pytest.mark.asyncio
    async def test_failed_author_request_with_no_book_id_does_nothing(self) -> None:
        """Verify failing an author request with no book_id short-circuits without popping or finalizing."""
        settings = make_settings()
        state = make_state(sample_size=1)
        state.pending_author_records["1"] = make_record(book_id="1")

        crawler = _build_crawlee_crawler(settings=settings)
        raw_handler = attach_and_capture_raw_failed_handler(
            crawler=crawler, state=state, settings=settings
        )

        context = build_failed_context(label=settings.label_author, book_id="1")
        context.request.user_data = {}

        result = await raw_handler(context, Exception("boom"))

        assert result is None
        assert "1" in state.pending_author_records
        assert state.successes == 0

    @pytest.mark.asyncio
    async def test_failed_author_request_with_no_pending_record_does_nothing(self) -> None:
        """Verify failing an author request without a matching pending record exits cleanly without errors."""
        settings = make_settings()
        state = make_state(sample_size=1)

        crawler = _build_crawlee_crawler(settings=settings)
        raw_handler = attach_and_capture_raw_failed_handler(
            crawler=crawler, state=state, settings=settings
        )

        context = build_failed_context(label=settings.label_author, book_id="unknown")
        result = await raw_handler(context, Exception("boom"))

        assert result is None
        assert state.successes == 0

    @pytest.mark.asyncio
    async def test_unrecognized_label_falls_through_without_side_effects(self) -> None:
        """Verify failure handler gracefully ignores requests with unrecognized route labels."""
        settings = make_settings()
        state = make_state(sample_size=1)

        crawler = _build_crawlee_crawler(settings=settings)
        raw_handler = attach_and_capture_raw_failed_handler(
            crawler=crawler, state=state, settings=settings
        )

        context = build_failed_context(label="SOME_OTHER_LABEL", book_id="1")
        result = await raw_handler(context, Exception("boom"))

        assert result is None
        context.add_requests.assert_not_called()
        assert state.successes == 0


# ---------------------------------------------------------------------------
# _drive_crawl_to_completion
# ---------------------------------------------------------------------------


class TestDriveCrawlToCompletion:
    """Tests evaluating asynchronous crawl loop execution and event completion races."""

    @pytest.mark.asyncio
    async def test_waits_for_run_task_when_it_finishes_first(self) -> None:
        """Verify drive process completes naturally when the crawler run task finishes prior to done event."""
        state = make_state(sample_size=5)
        crawler = MagicMock()
        crawler.run = AsyncMock(return_value=None)

        await _drive_crawl_to_completion(crawler=crawler, seed_requests=[], state=state)
        crawler.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancels_run_task_when_done_event_fires_first(self) -> None:
        """Verify active crawler run task is cancelled promptly once `done_event` is signaled."""
        state = make_state(sample_size=1)

        async def slow_run(requests: list[Any]) -> None:
            await asyncio.sleep(10)

        crawler = MagicMock()
        crawler.run = AsyncMock(side_effect=slow_run)

        async def set_done_soon() -> None:
            await asyncio.sleep(0.01)
            state.done_event.set()

        asyncio.ensure_future(set_done_soon())
        await _drive_crawl_to_completion(crawler=crawler, seed_requests=[], state=state)


# ---------------------------------------------------------------------------
# _reset_request_queue_storage
# ---------------------------------------------------------------------------


class TestResetRequestQueueStorage:
    """Tests evaluating the Crawlee request-queue storage reset helper."""

    @pytest.mark.asyncio
    async def test_opens_and_drops_default_request_queue(self, mocker: Any) -> None:
        """Verify the helper opens the default queue and drops it exactly once."""
        mock_queue = mocker.AsyncMock()
        mock_open = mocker.patch(
            "src.goodreads_etl.extractors.extract.RequestQueue.open",
            new_callable=mocker.AsyncMock,
            return_value=mock_queue,
        )

        await _reset_request_queue_storage()

        mock_open.assert_awaited_once_with()
        mock_queue.drop.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_propagates_exception_from_open(self, mocker: Any) -> None:
        """Verify an exception raised while opening the queue propagates to the caller."""
        mocker.patch(
            "src.goodreads_etl.extractors.extract.RequestQueue.open",
            new_callable=mocker.AsyncMock,
            side_effect=RuntimeError("storage backend unavailable"),
        )

        with pytest.raises(RuntimeError, match="storage backend unavailable"):
            await _reset_request_queue_storage()

    @pytest.mark.asyncio
    async def test_propagates_exception_from_drop(self, mocker: Any) -> None:
        """Verify an exception raised while dropping the queue propagates to the caller."""
        mock_queue = mocker.AsyncMock()
        mock_queue.drop.side_effect = RuntimeError("drop failed")
        mocker.patch(
            "src.goodreads_etl.extractors.extract.RequestQueue.open",
            new_callable=mocker.AsyncMock,
            return_value=mock_queue,
        )

        with pytest.raises(RuntimeError, match="drop failed"):
            await _reset_request_queue_storage()

    @pytest.mark.asyncio
    async def test_is_called_by_run_sampling_crawl(self, mocker: Any) -> None:
        """Verify run_sampling_crawl invokes the reset helper before seeding requests."""
        reset_mock = mocker.patch(
            "src.goodreads_etl.extractors.extract._reset_request_queue_storage",
            new_callable=mocker.AsyncMock,
        )
        mocker.patch(
            "src.goodreads_etl.extractors.extract._seed_initial_requests",
            return_value=None,
        )

        await run_sampling_crawl(
            sample_size=5, min_book_id=1, max_book_id=100, settings=make_settings()
        )

        reset_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_not_called_when_sample_size_non_positive(self, mocker: Any) -> None:
        """Verify the reset helper is skipped entirely for non-positive sample sizes."""
        reset_mock = mocker.patch(
            "src.goodreads_etl.extractors.extract._reset_request_queue_storage",
            new_callable=mocker.AsyncMock,
        )

        await run_sampling_crawl(
            sample_size=0, min_book_id=1, max_book_id=100, settings=make_settings()
        )

        reset_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# run_sampling_crawl -- top-level orchestration
# ---------------------------------------------------------------------------


class TestRunSamplingCrawl:
    """Tests evaluating entrypoint sampling orchestration logic."""

    @pytest.fixture(autouse=True)
    def _mock_storage_reset(self, mocker: Any) -> None:
        """Prevent real Crawlee file-system I/O from slowing down every test."""
        mocker.patch(
            "src.goodreads_etl.extractors.extract._reset_request_queue_storage",
            new_callable=AsyncMock,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sample_size", [0, -1])
    async def test_returns_empty_list_for_non_positive_sample_size(self, sample_size: int) -> None:
        """Verify entrypoint instantly returns empty list when `sample_size` is zero or negative."""
        result = await run_sampling_crawl(
            sample_size=sample_size, min_book_id=1, max_book_id=100, settings=make_settings()
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_seeding_fails(self, mocker: Any) -> None:
        """Verify entrypoint returns empty list if request seeding returns None."""
        mocker.patch(
            "src.goodreads_etl.extractors.extract._seed_initial_requests",
            return_value=None,
        )
        result = await run_sampling_crawl(
            sample_size=5, min_book_id=1, max_book_id=100, settings=make_settings()
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_already_tried_book_ids_are_preseeded_into_state(self, mocker: Any) -> None:
        """Verify pre-existing tried book IDs passed into orchestrator are registered into state tracker."""
        captured_state = {}

        async def fake_drive(*, crawler: Any, seed_requests: Any, state: Any) -> None:
            captured_state["state"] = state
            state.done_event.set()

        mocker.patch(
            "src.goodreads_etl.extractors.extract._drive_crawl_to_completion",
            side_effect=fake_drive,
        )
        await run_sampling_crawl(
            sample_size=1,
            min_book_id=1,
            max_book_id=1000,
            settings=make_settings(),
            already_tried_book_ids={"999"},
        )
        assert "999" in captured_state["state"].tried_book_ids

    @pytest.mark.asyncio
    async def test_returns_records_truncated_to_sample_size(self, mocker: Any) -> None:
        """Verify collected result array is strictly truncated to requested `sample_size` limit."""

        async def fake_drive(*, crawler: Any, seed_requests: Any, state: Any) -> None:
            state.records.extend([make_record(book_id=str(i)) for i in range(5)])

        mocker.patch(
            "src.goodreads_etl.extractors.extract._drive_crawl_to_completion",
            side_effect=fake_drive,
        )
        result = await run_sampling_crawl(
            sample_size=2, min_book_id=1, max_book_id=1000, settings=make_settings()
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_logs_warning_when_crawl_ends_early(self, mocker: Any) -> None:
        """Verify warning log is emitted when crawl loop terminates before reaching full target sample size."""
        warning_mock = mocker.patch("src.goodreads_etl.extractors.extract.logger.warning")

        async def fake_drive(*, crawler: Any, seed_requests: Any, state: Any) -> None:
            state.records.append(make_record())

        mocker.patch(
            "src.goodreads_etl.extractors.extract._drive_crawl_to_completion",
            side_effect=fake_drive,
        )
        await run_sampling_crawl(
            sample_size=5, min_book_id=1, max_book_id=1000, settings=make_settings()
        )
        warning_mock.assert_called_once()
