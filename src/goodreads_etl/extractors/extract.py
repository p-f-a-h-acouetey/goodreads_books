"""EXTRACT stage: crawl Goodreads book/author/editions pages, assemble book records.

Owns everything needed to turn "collect N more books" into structured
records, in one class: request building, page parsing, per-run crawl
state, and the queue-based self-healing crawl itself.

Uses Crawlee's BeautifulSoupCrawler with a single persistent RequestQueue
per run -- one crawler.run(seed_requests) call processes the whole crawl,
not one .run() call per page/book. Book, author, and editions pages are
routed by label; any terminal outcome for a book (success, no usable
data, or a permanently failed request) immediately enqueues one
replacement book_id, keeping the queue self-healing without any local
retry loop.

Exposes exactly one class, BookExtractor, with one public method:
    - extract(target_count, tried_book_ids)
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from bs4 import BeautifulSoup
from crawlee import ConcurrencySettings, Request
from crawlee.crawlers import BasicCrawlingContext, BeautifulSoupCrawler, BeautifulSoupCrawlingContext
from crawlee.storages import RequestQueue
from loguru import logger

# --- Goodreads URL templates ------------------------------------------------
BASE_BOOK_URL = "https://www.goodreads.com/book/show/"
EDITIONS_URL_TEMPLATE = "https://www.goodreads.com/work/editions/{work_id}"

# --- Random book ID sampling -------------------------------------------------
MIN_BOOK_ID = 1  # Known
MAX_BOOK_ID = 10_000_000  # Unknown, just an arbitrary max value
MAX_RANDOM_ID_ATTEMPTS = 500  # Give up drawing a fresh ID after this many collisions

# --- Crawlee route labels -----------------------------------------------------
LABEL_BOOK = "book"
LABEL_AUTHOR = "author"
LABEL_EDITIONS = "editions"

# --- Crawler tuning -----------------------------------------------------------
MAX_REQUEST_RETRIES = 10
REQUEST_HANDLER_TIMEOUT_SECONDS = 60

# --- Concurrency & real request pacing -----------------------------------------
# desired_concurrency / max_concurrency govern LOCAL resource usage --
# Crawlee's AutoscaledPool scales toward max_concurrency based on CPU/memory/
# event-loop health, with NO awareness of how the remote site is responding.
# max_tasks_per_minute is the one setting that caps the REAL outgoing request
# rate against Goodreads, independent of local concurrency -- leaving it at
# its default of inf would let the pool scale toward max_concurrency almost
# immediately for a lightweight HTTP crawler, reproducing WAF throttling.
CONCURRENCY_SETTINGS = ConcurrencySettings(
    desired_concurrency=5,
    max_concurrency=10,
    max_tasks_per_minute=60,  # ~1 request/second, matching Goodreads' documented API tolerance
)

# --- Contributor stats text patterns -------------------------------------------
FOLLOWERS_PATTERN = re.compile(r"followers\s*\(([\d,]+)\)", re.I)
DISTINCT_WORKS_PATTERN = re.compile(r"([\d,]+)\s+distinct\s+works?", re.I)

# --- Catalog guideline violation -----------------------------------------------
CATALOG_GUIDELINE_PATTERN = re.compile(r"does\s+not\s+meet\s+our\s+catalog\s+guidelines", re.I)

# --- Editions count -------------------------------------------------------------
EDITIONS_HEADING_PATTERN = re.compile(r"\b([\d,]+)\s+editions?\b", re.I)
EDITIONS_TOTAL_PATTERN = re.compile(
    r"showing\s+(?:[\d,]+\s*-\s*[\d,]+\s+of\s+([\d,]+)|all\s+([\d,]+))",
    re.I,
)

# --- Apollo cache recursion -------------------------------------------------------
APOLLO_CACHE_MAX_DEPTH = 15  # generous ceiling against reference cycles; real payloads need ~8-10


class BookExtractor:
    """Drives one queue-based, self-healing sampling crawl per call, from raw HTML to final records."""

    @dataclass(slots=True)
    class _Progress:
        """Mutable state shared across all route handlers for one crawl run.

        Attributes:
            target_count: Number of NEW valid book records to collect this run.
            tried_book_ids: Set tracking every ID drawn so far (this run + all prior runs).
            records: List holding fully assembled book record dicts.
            pending: Maps book_id -> in-progress record, awaiting author/editions enrichment.
            pending_editions_lookup: Maps editions_url -> book_id, so the editions handler
                knows which pending record to update.
            done_event: Async event set once target_count records are collected.
        """

        target_count: int
        tried_book_ids: set[str]
        records: list[dict[str, Any]] = field(default_factory=list)
        pending: dict[str, dict[str, Any]] = field(default_factory=dict)
        pending_editions_lookup: dict[str, str] = field(default_factory=dict)
        done_event: asyncio.Event = field(default_factory=asyncio.Event)

        @property
        def successes(self) -> int:
            """Return count of successfully collected records this run.

            Returns:
                Number of records collected so far in this run.
            """
            return len(self.records)

        @property
        def is_complete(self) -> bool:
            """Check whether this run's target has been met.

            Returns:
                True if collected records meet or exceed target_count.
            """
            return self.successes >= self.target_count

        def next_random_book_id(self) -> str | None:
            """Draw, reserve, and return one previously untried book ID.

            Returns:
                A unique random book ID string, or None if no untried ID was
                found within MAX_RANDOM_ID_ATTEMPTS.
            """
            for _ in range(MAX_RANDOM_ID_ATTEMPTS):
                candidate = str(random.randint(MIN_BOOK_ID, MAX_BOOK_ID))
                if candidate not in self.tried_book_ids:
                    self.tried_book_ids.add(candidate)
                    return candidate
            return None

        def add_record(self, record: dict[str, Any]) -> bool:
            """Store a completed record if capacity remains; signal completion if target is met.

            Args:
                record: The completed book record dict to store.

            Returns:
                True if the record was added, False if the run is already complete.
            """
            if self.is_complete:
                return False
            self.records.append(record)
            if self.is_complete:
                self.done_event.set()
            return True

    def __init__(self) -> None:
        self.logger = logger.bind(component="BookExtractor")

    # --- Parsing (private -- only ever called from this class's own route handlers) ---

    def _ms_epoch_to_iso(self, ms: int | None) -> str | None:
        """Convert an Apollo-cache millisecond-epoch timestamp to an ISO date.

        Uses timedelta arithmetic from the Unix epoch rather than
        datetime.fromtimestamp, since the latter relies on the platform C
        runtime's localtime/gmtime functions -- on Windows specifically,
        this raises OSError: [Errno 22] Invalid argument for any date
        before January 2, 1970, which silently breaks every book first
        published before that year (a large fraction of any random sample
        across Goodreads' full catalog).

        Args:
            ms: Milliseconds since Unix epoch, or None/0 if missing.

        Returns:
            ISO 8601 date string, or None if ms is falsy.
        """
        if not ms:
            return None
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        return (epoch + timedelta(milliseconds=ms)).date().isoformat()

    def _resolve_refs(self, state: dict[str, Any], node: Any, depth: int = 0) -> Any:
        """Recursively replace Apollo cache {"__ref": "Type:id"} pointers with the object they point to.

        Args:
            state: The full flat Apollo cache dict (apolloState).
            node: The current node being resolved (dict, list, or scalar).
            depth: Current recursion depth; guards against reference cycles.

        Returns:
            node with every __ref pointer replaced, or None past APOLLO_CACHE_MAX_DEPTH.
        """
        if depth > APOLLO_CACHE_MAX_DEPTH:
            return None
        if isinstance(node, dict):
            if list(node.keys()) == ["__ref"]:
                return self._resolve_refs(state, state.get(node["__ref"]), depth + 1)
            return {key: self._resolve_refs(state, value, depth + 1) for key, value in node.items()}
        if isinstance(node, list):
            return [self._resolve_refs(state, item, depth + 1) for item in node]
        return node

    def _load_apollo_state(self, soup: BeautifulSoup) -> dict[str, Any] | None:
        """Parse and return the raw apolloState dict from a page's __NEXT_DATA__ script tag.

        Args:
            soup: Parsed HTML of any Goodreads page using Next.js hydration.

        Returns:
            The raw (un-de-referenced) apolloState dict, or None if absent/malformed.
        """
        tag = soup.find("script", id="__NEXT_DATA__")
        if not tag or not tag.string:
            return None
        try:
            payload = json.loads(tag.string)
            return payload["props"]["pageProps"]["apolloState"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _find_book_entry(self, apollo_state: dict[str, Any], book_id: str) -> dict | None:
        """Find and resolve the Apollo cache Book entry matching a given book_id.

        Args:
            apollo_state: The full flat Apollo cache dict.
            book_id: The Goodreads numeric book ID to match against.

        Returns:
            The resolved (de-referenced) Book object dict, or None if not found.
        """
        raw_key = next(
            (
                key
                for key, value in apollo_state.items()
                if key.startswith("Book:")
                and isinstance(value, dict)
                and str(value.get("legacyId")) == str(book_id)
            ),
            None,
        )
        if raw_key is None:
            return None
        return self._resolve_refs(apollo_state, apollo_state[raw_key])

    def _extract_all_contributors(self, book: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract every contributor (primary author + secondary contributors).

        Args:
            book: A resolved (de-referenced) Book object from the Apollo cache.

        Returns:
            List of dicts with name, url, and role for every contributor found.
        """
        contributors = []
        primary_edge = book.get("primaryContributorEdge") or {}
        primary_node = primary_edge.get("node") or {}
        if primary_node.get("name"):
            contributors.append({
                "name": primary_node.get("name"),
                "url": primary_node.get("webUrl"),
                "role": primary_edge.get("role") or "Author",
            })
        for edge in book.get("secondaryContributorEdges") or []:
            if not isinstance(edge, dict):
                continue
            node = edge.get("node") or {}
            name = node.get("name")
            if not name:
                continue
            contributors.append({"name": name, "url": node.get("webUrl"), "role": edge.get("role") or "Contributor"})
        return contributors

    def _parse_book_page(self, soup: BeautifulSoup, book_id: str) -> dict[str, Any]:
        """Parse a book page's __NEXT_DATA__ Apollo cache into a structured dict.

        editions_url is derived directly from the work's legacyId
        (deterministic Goodreads URL scheme), not the Apollo cache's
        work.editions.__ref -> webUrl chain, which is frequently
        unpopulated for books with few editions.

        Args:
            soup: Parsed HTML of a Goodreads book page.
            book_id: The Goodreads numeric book ID this page is for.

        Returns:
            Dict with every book field, editions_url, first_author_url,
            is_nonexistent, and catalog_guideline_violation.
        """
        page_text = soup.get_text(" ", strip=True)
        catalog_guideline_violation = bool(CATALOG_GUIDELINE_PATTERN.search(page_text))

        apollo_state = self._load_apollo_state(soup)
        book = self._find_book_entry(apollo_state, book_id) if apollo_state else None

        if book is None or not book.get("title"):
            return {
                "title": None, "contributors": [], "first_author": None, "first_author_url": None,
                "description": None, "genres": [], "series": [], "format": None, "pages": None,
                "language": None, "asin": None, "isbn10": None, "isbn13": None, "publisher": None,
                "publication_date": None, "average_rating": None, "ratings_count": None,
                "reviews_count": None, "editions_url": None, "work_id": None,
                "is_nonexistent": not catalog_guideline_violation,
                "catalog_guideline_violation": catalog_guideline_violation,
            }

        work = book.get("work") or {}
        details = book.get("details") or {}
        stats = work.get("stats") or {}
        work_id = work.get("legacyId")

        genres = [
            genre_edge["genre"]["name"]
            for genre_edge in (book.get("bookGenres") or [])
            if isinstance(genre_edge, dict) and genre_edge.get("genre", {}).get("name")
        ]
        series = [
            {"name": entry.get("series", {}).get("title"), "position": entry.get("userPosition")}
            for entry in (book.get("bookSeries") or [])
            if entry.get("series", {}).get("title")
        ]
        contributors = self._extract_all_contributors(book)
        first_author_url = contributors[0]["url"] if contributors else None

        description_html = book.get("description")
        description_text = (
            BeautifulSoup(description_html, "html.parser").get_text("\n").strip()
            if description_html
            else None
        )

        return {
            "title": book.get("title"),
            "contributors": contributors,
            "first_author": contributors[0]["name"] if contributors else None,
            "first_author_url": first_author_url,
            "description": description_text,
            "genres": genres,
            "series": series,
            "format": details.get("format"),
            "pages": details.get("numPages"),
            "language": (details.get("language") or {}).get("name"),
            "asin": details.get("asin"),
            "isbn10": details.get("isbn"),
            "isbn13": details.get("isbn13"),
            "publisher": details.get("publisher"),
            "publication_date": self._ms_epoch_to_iso(details.get("publicationTime")),
            "average_rating": stats.get("averageRating"),
            "ratings_count": stats.get("ratingsCount"),
            "reviews_count": stats.get("textReviewsCount"),
            "editions_url": EDITIONS_URL_TEMPLATE.format(work_id=work_id) if work_id else None,
            "work_id": work_id,
            "is_nonexistent": False,
            "catalog_guideline_violation": catalog_guideline_violation,
        }

    def _parse_author_page(self, soup: BeautifulSoup) -> dict[str, Any]:
        """Extract follower count and distinct-works count from an author page.

        Args:
            soup: Parsed HTML of a Goodreads author profile page.

        Returns:
            Dict with num_followers and num_distinct_works.
        """
        page_text = soup.get_text(" ", strip=True)
        followers_match = FOLLOWERS_PATTERN.search(page_text)
        works_match = DISTINCT_WORKS_PATTERN.search(page_text)
        return {
            "num_followers": int(followers_match.group(1).replace(",", "")) if followers_match else None,
            "num_distinct_works": int(works_match.group(1).replace(",", "")) if works_match else None,
        }

    def _parse_editions_page(self, soup: BeautifulSoup) -> int | None:
        """Extract the exact total edition count from an editions-listing page.

        Args:
            soup: Parsed HTML of a work's editions-listing page (first page).

        Returns:
            The exact total number of editions, or None if not found.
        """
        page_text = soup.get_text(" ", strip=True)
        match = EDITIONS_TOTAL_PATTERN.search(page_text)
        if match:
            return int((match.group(1) or match.group(2)).replace(",", ""))
        match = EDITIONS_HEADING_PATTERN.search(page_text)
        if match:
            return int(match.group(1).replace(",", ""))
        return None

    # --- Request building (private) ---

    def _build_book_request(self, book_id: str) -> Request:
        """Build a Crawlee request for a single book page.

        Args:
            book_id: The Goodreads book ID to target.

        Returns:
            Configured Request object for the book page, labeled LABEL_BOOK.
        """
        return Request.from_url(
            f"{BASE_BOOK_URL}{book_id}",
            label=LABEL_BOOK,
            user_data={"book_id": book_id},
            unique_key=f"book-{book_id}",
        )

    def _build_author_request(self, author_url: str, book_id: str) -> Request:
        """Build a Crawlee request for a book's first author page.

        Args:
            author_url: The full profile URL of the author.
            book_id: Associated book_id awaiting author enrichment.

        Returns:
            Configured Request object for the author page, labeled LABEL_AUTHOR.
        """
        return Request.from_url(
            author_url,
            label=LABEL_AUTHOR,
            user_data={"book_id": book_id},
            unique_key=f"author-{book_id}",
        )

    def _build_editions_request(self, editions_url: str, book_id: str) -> Request:
        """Build a Crawlee request for a work's editions-listing page.

        Args:
            editions_url: The full editions-listing URL for the book's work.
            book_id: Associated book_id awaiting editions-count enrichment.

        Returns:
            Configured Request object for the editions page, labeled LABEL_EDITIONS.
        """
        return Request.from_url(
            editions_url,
            label=LABEL_EDITIONS,
            user_data={"book_id": book_id},
            unique_key=f"editions-{book_id}",
        )

    def _require_book_id(self, request: Request) -> str:
        """Extract and validate book_id from a request's user_data.

        Args:
            request: The Crawlee request carrying book_id in its user_data.

        Returns:
            The validated string book_id.

        Raises:
            ValueError: If book_id is missing or not a string.
        """
        book_id = request.user_data.get("book_id")
        if not isinstance(book_id, str):
            raise ValueError(f"Missing or invalid book_id in request user_data: {book_id!r}")
        return book_id

    def _optional_book_id(self, request: Request) -> str | None:
        """Extract book_id from a request's user_data, if valid.

        Args:
            request: The Crawlee request carrying book_id in its user_data.

        Returns:
            The string book_id if present and valid, otherwise None.
        """
        book_id = request.user_data.get("book_id")
        return book_id if isinstance(book_id, str) else None

    # --- Record assembly (private) ---

    def _start_record(self, book_id: str, book_data: dict[str, Any]) -> dict[str, Any]:
        """Build the initial pending record for a book from its parsed book page.

        Tracks which enrichment steps (author, editions) are still
        outstanding via internal "_awaiting_*" flags, consumed by
        _is_ready_to_assemble and stripped by _assemble_record.

        Args:
            book_id: The Goodreads numeric book ID this record is for.
            book_data: The dict returned by _parse_book_page for this book.

        Returns:
            A pending record dict, not yet finalized.
        """
        record = {key: value for key, value in book_data.items() if key not in ("is_nonexistent", "first_author_url", "editions_url")}
        record["book_id"] = book_id
        record["url"] = f"{BASE_BOOK_URL}{book_id}"
        record["num_editions"] = None
        record["has_more_editions"] = False
        record["_awaiting_author"] = bool(book_data.get("first_author_url"))
        record["_awaiting_editions"] = bool(book_data.get("editions_url"))
        return record

    def _is_ready_to_assemble(self, record: dict[str, Any]) -> bool:
        """Check whether a pending record has no more enrichment steps outstanding.

        Args:
            record: The pending record dict to check.

        Returns:
            True if neither author nor editions enrichment is still pending.
        """
        return not record.get("_awaiting_author") and not record.get("_awaiting_editions")

    def _assemble_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Strip internal bookkeeping flags and return the final record, ready to persist.

        Args:
            record: A pending record dict that has passed _is_ready_to_assemble.

        Returns:
            The final record dict, with derived fields set and no
            leading-underscore internal keys remaining.
        """
        record.pop("_awaiting_author", None)
        record.pop("_awaiting_editions", None)
        record["is_part_of_series"] = bool(record.get("series"))
        return record

    # --- Crawl construction and routing (private) ---

    def _build_crawler(self) -> BeautifulSoupCrawler:
        """Construct a bare BeautifulSoupCrawler with global concurrency settings applied.

        Returns:
            An un-routed BeautifulSoupCrawler instance.
        """
        return BeautifulSoupCrawler(
            max_request_retries=MAX_REQUEST_RETRIES,
            request_handler_timeout=timedelta(seconds=REQUEST_HANDLER_TIMEOUT_SECONDS),
            retry_on_blocked=True,
            concurrency_settings=CONCURRENCY_SETTINGS,
            configure_logging=False,
        )

    def _attach_handlers(self, crawler: BeautifulSoupCrawler, progress: "BookExtractor._Progress") -> None:
        """Wire book/author/editions/failure handlers onto crawler, closing over progress.

        Args:
            crawler: The BeautifulSoupCrawler instance to attach routes to.
            progress: Shared mutable _Progress tracking this run's state.
        """

        async def _enqueue_replacement(context: BeautifulSoupCrawlingContext) -> None:
            """Enqueue one fresh random book ID unless the run is already complete."""
            if progress.is_complete:
                return None
            new_id = progress.next_random_book_id()
            if new_id is None:
                self.logger.warning("No untried book ID found within attempt limit -- ID space may be exhausted")
                return None
            await context.add_requests([self._build_book_request(new_id)])

        async def _try_finalize(context: BeautifulSoupCrawlingContext, book_id: str) -> None:
            """Assemble and store a pending record once it has no more enrichment steps outstanding."""
            record = progress.pending.get(book_id)
            if record is None or not self._is_ready_to_assemble(record):
                return None
            progress.pending.pop(book_id)
            final_record = self._assemble_record(record)
            if not progress.add_record(final_record):
                return None
            self.logger.info(f"book_id={book_id}: collected ({progress.successes}/{progress.target_count} target)")
            if not progress.is_complete:
                await _enqueue_replacement(context)

        @crawler.router.handler(LABEL_BOOK)
        async def handle_book(context: BeautifulSoupCrawlingContext) -> None:
            """Route handler for Goodreads book pages."""
            if progress.done_event.is_set():
                return None

            book_id = self._require_book_id(context.request)
            book_data = self._parse_book_page(context.soup, book_id)

            if book_data["is_nonexistent"]:
                await _enqueue_replacement(context)
                return None

            if progress.done_event.is_set():
                return None

            record = self._start_record(book_id, book_data)
            progress.pending[book_id] = record

            editions_url = book_data.get("editions_url")
            first_author_url = book_data.get("first_author_url")
            if editions_url:
                progress.pending_editions_lookup[editions_url] = book_id
                await context.add_requests([self._build_editions_request(editions_url, book_id)])
            if first_author_url:
                await context.add_requests([self._build_author_request(first_author_url, book_id)])

            await _try_finalize(context, book_id)

        @crawler.router.handler(LABEL_AUTHOR)
        async def handle_author(context: BeautifulSoupCrawlingContext) -> None:
            """Route handler for Goodreads author profile pages."""
            book_id = self._require_book_id(context.request)
            record = progress.pending.get(book_id)
            if record is None:
                return None

            author_stats = self._parse_author_page(context.soup)
            if record.get("contributors"):
                record["contributors"][0] = {**record["contributors"][0], **author_stats}
            record["_awaiting_author"] = False

            await _try_finalize(context, book_id)

        @crawler.router.handler(LABEL_EDITIONS)
        async def handle_editions(context: BeautifulSoupCrawlingContext) -> None:
            """Route handler for Goodreads work editions-listing pages."""
            book_id = self._require_book_id(context.request)
            progress.pending_editions_lookup.pop(context.request.url, None)

            record = progress.pending.get(book_id)
            if record is None:
                return None

            num_editions = self._parse_editions_page(context.soup)
            record["num_editions"] = num_editions
            record["has_more_editions"] = bool(num_editions and num_editions > 1)
            record["_awaiting_editions"] = False

            await _try_finalize(context, book_id)

        @crawler.failed_request_handler
        async def handle_failed(context: BeautifulSoupCrawlingContext | BasicCrawlingContext, error: Exception) -> None:
            """Handler for permanently failed Crawlee requests (e.g. exhausted retries, blocked)."""
            label = context.request.label
            book_id = self._optional_book_id(context.request)
            self.logger.warning(f"Request failed permanently: label={label} book_id={book_id} error={error}")

            bs_context : BeautifulSoupCrawlingContext | BasicCrawlingContext = context 

            if label == LABEL_BOOK:
                if book_id is not None:
                    progress.pending.pop(book_id, None)
                await _enqueue_replacement(bs_context) # type: ignore
                return None

            if book_id is None:
                return None

            record = progress.pending.get(book_id)
            if record is None:
                return None

            if label == LABEL_AUTHOR:
                record["_awaiting_author"] = False
            elif label == LABEL_EDITIONS:
                progress.pending_editions_lookup.pop(context.request.url, None)
                record["_awaiting_editions"] = False

            await _try_finalize(bs_context, book_id) # type: ignore

    async def _drive_to_completion(self, crawler: BeautifulSoupCrawler, seed_requests: list[Request], progress: "BookExtractor._Progress") -> None:
        """Run the crawler until naturally finished or the target is met, whichever first.

        Args:
            crawler: Configured BeautifulSoupCrawler instance.
            seed_requests: Initial batch of requests to seed the queue with.
            progress: Shared mutable _Progress containing target/done_event.
        """
        run_task = asyncio.create_task(crawler.run(seed_requests))
        done_wait_task = asyncio.create_task(progress.done_event.wait())

        done, pending = await asyncio.wait({run_task, done_wait_task}, return_when=asyncio.FIRST_COMPLETED)

        if done_wait_task in done and not run_task.done():
            run_task.cancel()
        for task in pending:
            task.cancel()

        try:
            await run_task
        except asyncio.CancelledError:
            pass

    async def _reset_request_queue_storage(self) -> None:
        """Drop any request-queue state left over from a previous asyncio event loop.

        Crawlee's file-system request queue client caches an asyncio.Lock
        bound to whatever event loop was active when first created. Since
        each notebook cell execution can run its own event loop, that lock
        can end up bound to a dead loop on repeated runs; dropping the
        queue forces a fresh client/lock on the next open().
        """
        queue = await RequestQueue.open()
        await queue.drop()

    async def extract(self, target_count: int, tried_book_ids: set[str]) -> list[dict[str, Any]]:
        """Run a queue-based sampling crawl until target_count new valid records are collected.

        This is the single public entry point BookRunner calls. Seeds
        desired_concurrency * 4 initial book requests (capped at
        target_count) so multiple books are in flight from the start,
        then lets Crawlee's own AutoscaledPool (bounded by
        CONCURRENCY_SETTINGS.max_tasks_per_minute) and the self-healing
        replacement handlers keep the queue fed until either the target
        is met or the queue naturally empties (ID space exhausted).

        Args:
            target_count: Number of NEW valid book records to collect.
            tried_book_ids: Set of book_ids to exclude (already scraped in
                prior runs); mutated in place with every ID drawn this run.

        Returns:
            List of up to target_count valid book record dicts.
        """
        if target_count <= 0:
            return []

        await self._reset_request_queue_storage()

        progress = self._Progress(target_count=target_count, tried_book_ids=tried_book_ids)

        seed_count = min(max(CONCURRENCY_SETTINGS.desired_concurrency * 4, 1), target_count)
        seed_ids = [progress.next_random_book_id() for _ in range(seed_count)]
        seed_requests = [self._build_book_request(book_id) for book_id in seed_ids if book_id is not None]
        if not seed_requests:
            self.logger.warning("Unable to seed crawl -- no untried book IDs available")
            return []

        crawler = self._build_crawler()
        self._attach_handlers(crawler, progress)

        await self._drive_to_completion(crawler, seed_requests, progress)

        if progress.successes < progress.target_count:
            self.logger.warning(
                f"Crawl ended early: {progress.successes}/{progress.target_count} collected "
                "(queue exhausted or ID space too sparse)"
            )

        return progress.records[:target_count]
