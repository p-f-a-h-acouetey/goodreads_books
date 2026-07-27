"""EXTRACT stage: crawl Goodreads book/author pages, assemble BookRecords.

Uses Crawlee's BeautifulSoupCrawler rather than a hand-rolled requests/retry
loop, so retries, concurrency, and session/proxy rotation come for free.
Book and author requests are routed by label; a book page that fails to
yield usable JSON-LD triggers a fresh random ID, keeping the sampling loop
self-healing without any local retry logic of our own.

Exposes exactly one public function: run_sampling_crawl.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, cast

from bs4 import BeautifulSoup
from crawlee import ConcurrencySettings, Request
from crawlee.crawlers import (
    BasicCrawlingContext,
    BeautifulSoupCrawler,
    BeautifulSoupCrawlingContext,
)
from crawlee.sessions import SessionPool
from loguru import logger

from src.goodreads_etl.utils.book_recorder import BookRecord
from src.goodreads_etl.utils.config_setter import SETTINGS, Settings
from src.goodreads_etl.utils.id_sampler import generate_random_book_id
from src.goodreads_etl.utils.page_parser import (
    DEFAULT_AGGREGATE_RATING_KEY,
    extract_author_stats_from_author_page,
    extract_description,
    extract_first_author_and_url,
    extract_first_published,
    extract_format,
    extract_genres,
    extract_json_ld,
    extract_next_data,
    extract_num_pages,
    extract_num_reviews,
    extract_publisher,
    extract_reading_stats,
    extract_series,
)
from src.goodreads_etl.utils.text_parser import clean_object, get_float

DEFAULT_RATING_VALUE_KEY = "ratingValue"
DEFAULT_NAME_KEY = "name"
DEFAULT_LANGUAGE_KEY = "inLanguage"


# ---------------------------------------------------------------------------
# Crawl-scoped mutable state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _SamplingState:
    """Mutable state shared across all handlers for one crawl run.

    Attributes:
        sample_size: Target number of valid book records to collect.
        min_book_id: Lower bound (inclusive) of book ID range.
        max_book_id: Upper bound (inclusive) of book ID range.
        max_random_id_attempts: Max attempts to pick an untried ID.
        tried_book_ids: Set tracking every ID drawn so far, successful or not.
        records: List holding fully collected `BookRecord` instances.
        pending_author_records: Map of `book_id` to `BookRecord` waiting for author page enrichment.
        done_event: Async event set when `sample_size` records are collected.
    """

    sample_size: int
    min_book_id: int
    max_book_id: int
    max_random_id_attempts: int

    tried_book_ids: set[str] = field(default_factory=set)
    records: list[BookRecord] = field(default_factory=list)
    pending_author_records: dict[str, BookRecord] = field(default_factory=dict)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def successes(self) -> int:
        """Return total count of successfully collected book records.

        Returns:
            Number of records collected.
        """
        return len(self.records)

    @property
    def is_complete(self) -> bool:
        """Check whether the sampling goal has been met.

        Returns:
            True if collected records meet or exceed sample size, else False.
        """
        return self.successes >= self.sample_size

    def next_random_book_id(self) -> str:
        """Draw, reserve, and return one previously untried book ID.

        Returns:
            A unique random book ID string.

        Raises:
            RuntimeError: If no untried ID can be found within attempt limit.
        """
        book_id = generate_random_book_id(
            min_book_id=self.min_book_id,
            max_book_id=self.max_book_id,
            excluded_book_ids=self.tried_book_ids,
            max_attempts=self.max_random_id_attempts,
        )
        self.tried_book_ids.add(book_id)
        return book_id

    def add_record(self, *, record: BookRecord) -> bool:
        """Store a completed record if capacity remains; signal completion.

        Args:
            record: The completed `BookRecord` instance to store.

        Returns:
            True if the record was successfully added, False if crawl is already complete.
        """
        if self.is_complete:
            return False
        self.records.append(record)
        if self.is_complete:
            self.done_event.set()
        return True


# ---------------------------------------------------------------------------
# User-data helpers (Crawlee's user_data is JsonSerializable, not `str`)
# ---------------------------------------------------------------------------


def _require_book_id(*, request: Request) -> str:
    """Extract and validate ``book_id`` from a request's user_data.

    Args:
        request: The Crawlee request carrying `book_id` in its user_data.

    Returns:
        The validated string book ID.

    Raises:
        ValueError: If `book_id` is missing or not a string.
    """
    book_id = request.user_data.get("book_id")
    if not isinstance(book_id, str):
        raise ValueError(f"Missing or invalid book_id in request user_data: {book_id!r}")
    return book_id


def _optional_book_id(*, request: Request) -> str | None:
    """Extract ``book_id`` from a request's user_data, if valid.

    Args:
        request: The Crawlee request carrying `book_id` in its user_data.

    Returns:
        The string book ID if present and valid, otherwise None.
    """
    book_id = request.user_data.get("book_id")
    return book_id if isinstance(book_id, str) else None


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def _build_book_request(*, book_id: str, settings: Settings) -> Request:
    """Build a Crawlee request for a single book page.

    Args:
        book_id: The Goodreads book ID to target.
        settings: Global application settings instance.

    Returns:
        Configured `Request` object for the book page.
    """
    return Request.from_url(
        f"{settings.base_book_url}{book_id}",
        label=settings.label_book,
        user_data={"book_id": book_id},
        unique_key=f"book-{book_id}",
    )


def _build_author_request(*, author_url: str, book_id: str, settings: Settings) -> Request:
    """Build a Crawlee request for a book's first author page.

    Args:
        author_url: The full profile URL of the author.
        book_id: Associated book ID awaiting author metadata.
        settings: Global application settings instance.

    Returns:
        Configured `Request` object for the author profile.
    """
    return Request.from_url(
        author_url,
        label=settings.label_author,
        user_data={"book_id": book_id},
        unique_key=f"author-{book_id}",
    )


# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------


def _build_record_from_book_page(
    *,
    book_id: str,
    soup: BeautifulSoup,
    json_ld: dict[str, Any],
    next_data: dict[str, Any] | None,
    settings: Settings,
) -> tuple[BookRecord, str | None]:
    """Assemble a BookRecord from one book page's soup and JSON-LD.

    Args:
        book_id: Unique string identifier for the book.
        soup: Parsed BeautifulSoup document of the book page.
        json_ld: Extracted JSON-LD structured data block.
        next_data: Extracted Next.js hydration payload (if present).
        settings: Global application settings instance.

    Returns:
        A tuple of `(assembled_book_record, first_author_url)`.
    """
    aggregate = json_ld.get(DEFAULT_AGGREGATE_RATING_KEY, {}) or {}
    html_text = str(soup)

    num_currently_reading, num_want_to_read = extract_reading_stats(soup=soup, html_text=html_text)
    first_author, first_author_url = extract_first_author_and_url(soup=soup, json_ld=json_ld)

    record = BookRecord(
        book_id=book_id,
        url=f"{settings.base_book_url}{book_id}",
        title=clean_object(value=json_ld.get(DEFAULT_NAME_KEY, "")) or "",
        first_author=first_author or "",
        first_author_url=first_author_url or "",
        average_rating=get_float(text=str(aggregate.get(DEFAULT_RATING_VALUE_KEY, ""))) or 0.0,
        num_reviews=extract_num_reviews(soup=soup, json_ld=json_ld) or 0,
        first_published=extract_first_published(soup=soup, json_ld=json_ld),
        publisher=extract_publisher(soup=soup, json_ld=json_ld, next_data=next_data),
        language_code=clean_object(value=str(json_ld.get(DEFAULT_LANGUAGE_KEY, ""))),
        num_pages=extract_num_pages(soup=soup, json_ld=json_ld) or 0,
        description=extract_description(soup=soup),
        genres=extract_genres(soup=soup),
        format=extract_format(soup=soup),
        series=extract_series(soup=soup),
        num_currently_reading=num_currently_reading,
        num_want_to_read=num_want_to_read,
    )

    return record, first_author_url


# ---------------------------------------------------------------------------
# Crawler construction
# ---------------------------------------------------------------------------


def _build_crawlee_crawler(*, settings: Settings) -> BeautifulSoupCrawler:
    """Construct a bare BeautifulSoupCrawler with settings applied.

    Handlers are attached separately in `_attach_handlers`, keeping crawler
    construction and behavior wiring as two distinct, testable steps.

    Args:
        settings: Global application settings instance.

    Returns:
        An un-routed `BeautifulSoupCrawler` instance.
    """
    kwargs: dict[str, Any] = {
        "max_request_retries": settings.num_retries,
        "request_handler_timeout": timedelta(seconds=settings.timeout_seconds),
        "use_session_pool": settings.use_session_pool,
        "retry_on_blocked": settings.retry_on_blocked,
        "additional_http_error_status_codes": list(settings.additional_http_error_status_codes),
        "max_session_rotations": settings.max_session_rotations,
    }

    if settings.max_concurrency not in (None, 0):
        kwargs["concurrency_settings"] = ConcurrencySettings(
            max_concurrency=settings.max_concurrency,
            desired_concurrency=settings.max_concurrency,
            max_tasks_per_minute=settings.max_tasks_per_minute,
        )

    if settings.use_session_pool:
        kwargs["session_pool"] = SessionPool(max_pool_size=settings.max_session_pool_size)

    return BeautifulSoupCrawler(**kwargs)


def _attach_handlers(
    *,
    crawler: BeautifulSoupCrawler,
    state: _SamplingState,
    settings: Settings,
) -> None:
    """Wire book/author/failure handlers onto ``crawler``.

    Closes over ``state`` so each handler can read/mutate the shared
    sampling progress.

    Args:
        crawler: The `BeautifulSoupCrawler` instance to attach routes to.
        state: Shared mutable `_SamplingState` tracking progress.
        settings: Global application settings instance.
    """

    async def _enqueue_replacement(context: BeautifulSoupCrawlingContext) -> None:
        """Enqueue one fresh random book ID unless the crawl is done.

        Args:
            context: Current Crawlee execution context.
        """
        if state.is_complete:
            return
        try:
            new_id = state.next_random_book_id()
        except RuntimeError as exc:
            logger.warning("No replacement book ID available: {}", exc)
            return
        await context.add_requests([_build_book_request(book_id=new_id, settings=settings)])

    async def _finalize_record(
        context: BeautifulSoupCrawlingContext, *, record: BookRecord
    ) -> None:
        """Persist a completed record and keep the sampling pipeline fed.

        Args:
            context: Current Crawlee execution context.
            record: Assembled `BookRecord` to persist.
        """
        if not state.add_record(record=record):
            return

        logger.info(
            "Scraped book_id={} ({}/{})", record.book_id, state.successes, state.sample_size
        )

        if not state.is_complete:
            await _enqueue_replacement(context)

    @crawler.router.handler(settings.label_book)
    async def handle_book(context: BeautifulSoupCrawlingContext) -> None:
        """Route handler for Goodreads book pages.

        Extracts metadata via JSON-LD, builds base record, and enqueues author
        enrichment or finalizes record directly.

        Args:
            context: Crawlee context containing the book page DOM.
        """
        if state.done_event.is_set():
            return

        book_id = _require_book_id(request=context.request)
        soup = context.soup

        json_ld = extract_json_ld(soup=soup)
        if not json_ld:
            logger.warning("No JSON-LD metadata found for book_id={}", book_id)
            await _enqueue_replacement(context)
            return

        next_data = extract_next_data(soup=soup)
        record, first_author_url = _build_record_from_book_page(
            book_id=book_id, soup=soup, json_ld=json_ld, next_data=next_data, settings=settings
        )

        if state.done_event.is_set():
            return

        if first_author_url:
            state.pending_author_records[book_id] = record
            await context.add_requests(
                [
                    _build_author_request(
                        author_url=first_author_url, book_id=book_id, settings=settings
                    )
                ]
            )
            return

        await _finalize_record(context, record=record)

    @crawler.router.handler(settings.label_author)
    async def handle_author(context: BeautifulSoupCrawlingContext) -> None:
        """Route handler for Goodreads author profile pages.

        Extracts author statistics and merges them into the pending `BookRecord`.

        Args:
            context: Crawlee context containing the author profile DOM.
        """
        book_id = _require_book_id(request=context.request)
        record = state.pending_author_records.pop(book_id, None)
        if record is None:
            return

        num_books, num_followers = extract_author_stats_from_author_page(
            html_text=str(context.soup)
        )
        record.first_author_num_books = num_books or 1  # At least 1 book which is the current one
        record.first_author_num_followers = num_followers or 0

        await _finalize_record(context, record=record)

    @crawler.failed_request_handler
    async def handle_failed(
        context: BeautifulSoupCrawlingContext | BasicCrawlingContext, error: Exception
    ) -> None:
        """Handler for permanently failed Crawlee requests.

        Recovers gracefully by enqueuing a new book ID or saving partial records.

        Args:
            context: Crawlee context of failed request.
            error: Caught exception leading to failure.
        """
        label = context.request.label
        book_id = _optional_book_id(request=context.request)
        logger.warning(
            "Request failed permanently label={} book_id={} error={}", label, book_id, error
        )

        bs_context = cast(BeautifulSoupCrawlingContext, context)

        if label == settings.label_book:
            if book_id is not None:
                state.pending_author_records.pop(book_id, None)
            await _enqueue_replacement(bs_context)
        elif label == settings.label_author and book_id is not None:
            record = state.pending_author_records.pop(book_id, None)
            if record is not None:
                await _finalize_record(bs_context, record=record)


def _seed_initial_requests(*, state: _SamplingState, settings: Settings) -> list[Request] | None:
    """Draw the first batch of book IDs to kick off the crawl.

    Args:
        state: Shared mutable `_SamplingState` tracking progress.
        settings: Global application settings instance.

    Returns:
        List of seed `Request` instances, or None if space exhaustion occurs.
    """
    seed_parallelism = settings.max_concurrency if settings.max_concurrency > 0 else 1
    seed_count = min(max(seed_parallelism * 2, 1), state.sample_size)

    try:
        return [
            _build_book_request(book_id=state.next_random_book_id(), settings=settings)
            for _ in range(seed_count)
        ]
    except RuntimeError as exc:
        logger.warning("Unable to seed crawl: {}", exc)
        return None


async def _drive_crawl_to_completion(
    *,
    crawler: BeautifulSoupCrawler,
    seed_requests: list[Request],
    state: _SamplingState,
) -> None:
    """Run the crawler until naturally finished or the target sample is met.

    Args:
        crawler: Configured `BeautifulSoupCrawler` instance.
        seed_requests: List of initial seeding requests.
        state: Shared mutable `_SamplingState` containing target details.
    """
    run_task = asyncio.create_task(crawler.run(seed_requests))
    done_wait_task = asyncio.create_task(state.done_event.wait())

    done, pending = await asyncio.wait(
        {run_task, done_wait_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if done_wait_task in done and not run_task.done():
        run_task.cancel()

    for task in pending:
        task.cancel()

    try:
        await run_task
    except asyncio.CancelledError:
        pass


async def run_sampling_crawl(
    *,
    sample_size: int,
    min_book_id: int,
    max_book_id: int,
    settings: Settings = SETTINGS,
    already_tried_book_ids: set[str] | None = None,
) -> list[BookRecord]:
    """Run a sampling crawl until ``sample_size`` successful records exist.

    This is the single public entrypoint for the extract stage.

    Args:
        sample_size: Desired number of valid `BookRecord` instances.
        min_book_id: Minimum numerical range for ID generation.
        max_book_id: Maximum numerical range for ID generation.
        settings: Application settings override instance.
        already_tried_book_ids: Optional set of IDs previously attempted to avoid duplicating.

    Returns:
        List of extracted `BookRecord` instances (up to `sample_size`).
    """
    if sample_size <= 0:
        return []

    state = _SamplingState(
        sample_size=sample_size,
        min_book_id=min_book_id,
        max_book_id=max_book_id,
        max_random_id_attempts=settings.max_random_id_attempts,
    )
    if already_tried_book_ids:
        state.tried_book_ids.update(already_tried_book_ids)

    seed_requests = _seed_initial_requests(state=state, settings=settings)
    if seed_requests is None:
        return []

    crawler = _build_crawlee_crawler(settings=settings)
    _attach_handlers(crawler=crawler, state=state, settings=settings)

    await _drive_crawl_to_completion(crawler=crawler, seed_requests=seed_requests, state=state)

    if state.successes < state.sample_size:
        logger.warning(
            "Crawl batch ended early: {}/{} successes (queue exhausted or ID space too sparse).",
            state.successes,
            state.sample_size,
        )

    return state.records[:sample_size]
