"""Central operational configuration, overridable via environment variables.

Only cross-cutting run/deploy knobs live here: crawl tuning, storage
locations, and pipeline defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial


def _read_env_str(*, name: str, default: str) -> str:
    """Read a string environment variable, falling back to a default.

    Args:
        name: The name of the environment variable to retrieve.
        default: The default string value to return if the environment
            variable is not set.

    Returns:
        The string value of the environment variable, or `default`
        if the environment variable is not found.
    """
    return os.getenv(name, default)


def _read_env_int(*, name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default.

    Args:
        name: The name of the environment variable to retrieve.
        default: The default integer value to return if the environment
            variable is not set.

    Returns:
        The integer value parsed from the environment variable, or `default`
        if the environment variable is not found.
    """
    raw_value = os.getenv(name)
    return int(raw_value) if raw_value is not None else default


def _read_env_bool(*, name: str, default: bool) -> bool:
    """Read a boolean environment variable, falling back to a default.

    Recognizes '1', 'true', 'yes', 'on' (case-insensitive) as True. Any other
    non-empty string evaluates to False.

    Args:
        name: The name of the environment variable to retrieve.
        default: The default boolean value to return if the environment
            variable is not set.

    Returns:
        True if the environment variable value matches a truthy string indicator,
        False if set to any other string value, or `default` if the environment
        variable is not found.
    """
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable, env-resolved settings for one pipeline run.

    Attributes:
        root_url: Target base web domain for scraping.
            Overridable via GOODREADS_ROOT_URL. Defaults to 'https://www.goodreads.com'.
        repo_id: Hugging Face dataset repository identifier.
            Overridable via HF_REPO_ID. Defaults to 'pfaha/goodreads-books'.
        encoding: Text file encoding standard used across IO operations. Defaults to 'utf-8'.
        book_ids_filename: Name of the text file tracking scraped book IDs.
            Defaults to 'scraped_book_ids.txt'.
        checkpoint_filename_template: Format string template for Parquet checkpoint files.
            Defaults to 'books-part{part}.parquet'.
        local_scraped_ids_path: Local file system path to the scraped IDs manifest.
            Overridable via LOCAL_SCRAPED_IDS_PATH. Defaults to 'data/scraped_book_ids.txt'.
        sample_size: Targeted total number of book records to collect per run.
            Overridable via SAMPLE_SIZE. Defaults to 100,000.
        checkpoint_every: Number of successful extractions before flushing a checkpoint file.
            Overridable via CHECKPOINT_EVERY. Defaults to 1000.
        min_book_id: Lower bound (inclusive) for random ID generation.
            Overridable via MIN_BOOK_ID. Defaults to 1.
        max_book_id: Upper bound (inclusive) for random ID generation.
            Overridable via MAX_BOOK_ID. Defaults to 60,000,000.
        max_random_id_attempts: Max draws per target item before raising an exhaustion error.
            Overridable via MAX_RANDOM_ID_ATTEMPTS. Defaults to 10,000.
        max_concurrency: Maximum number of concurrent request workers allowed.
            Overridable via MAX_CONCURRENCY. Defaults to 50.
        max_tasks_per_minute: Rate-limiting ceiling for outgoing HTTP tasks.
            Overridable via MAX_TASKS_PER_MINUTE. Defaults to 60.
        num_retries: Maximum automatic retry attempts allowed per HTTP failure.
            Overridable via NUM_RETRIES. Defaults to 3.
        timeout_seconds: Network request timeout threshold in seconds.
            Overridable via TIMEOUT_SECONDS. Defaults to 30.
        use_session_pool: Whether to maintain and reuse a pool of HTTP sessions.
            Overridable via USE_SESSION_POOL. Defaults to True.
        retry_on_blocked: Whether to treat blocking status codes (e.g., 429) as rotation triggers.
            Overridable via RETRY_ON_BLOCKED. Defaults to True.
        max_session_pool_size: Maximum capacity of active HTTP sessions in pool.
            Overridable via MAX_SESSION_POOL_SIZE. Defaults to 10.
        max_session_rotations: Maximum times a single worker session can be rotated.
            Overridable via MAX_SESSION_ROTATIONS. Defaults to 5.
        additional_http_error_status_codes: Tuple of HTTP status codes treated as blocking errors.
            Defaults to (429,).
        label_book: Scraping task router label for book detail pages. Defaults to 'BOOK'.
        label_author: Scraping task router label for author detail pages. Defaults to 'AUTHOR'.
        html_parser: Backend parser implementation used by HTML parsing utilities.
            Defaults to 'lxml'.
    """

    root_url: str = field(
        default_factory=partial(
            _read_env_str, name="GOODREADS_ROOT_URL", default="https://www.goodreads.com"
        )
    )
    repo_id: str = field(
        default_factory=partial(_read_env_str, name="HF_REPO_ID", default="pfaha/goodreads-books")
    )
    encoding: str = "utf-8"
    book_ids_filename: str = "scraped_book_ids.txt"
    checkpoint_filename_template: str = "books-part{part}.parquet"
    local_scraped_ids_path: str = field(
        default_factory=partial(
            _read_env_str, name="LOCAL_SCRAPED_IDS_PATH", default="data/scraped_book_ids.txt"
        )
    )

    sample_size: int = field(
        default_factory=partial(_read_env_int, name="SAMPLE_SIZE", default=100_000)
    )
    checkpoint_every: int = field(
        default_factory=partial(_read_env_int, name="CHECKPOINT_EVERY", default=1_000)
    )
    min_book_id: int = field(default_factory=partial(_read_env_int, name="MIN_BOOK_ID", default=1))
    # Goodreads book IDs currently top out well below 1e8; 60M is a safe
    # sampling ceiling that avoids wasting draws on IDs that never resolve.
    max_book_id: int = field(
        default_factory=partial(_read_env_int, name="MAX_BOOK_ID", default=60_000_000)
    )
    max_random_id_attempts: int = field(
        default_factory=partial(_read_env_int, name="MAX_RANDOM_ID_ATTEMPTS", default=10_000)
    )

    max_concurrency: int = field(
        default_factory=partial(_read_env_int, name="MAX_CONCURRENCY", default=50)
    )
    max_tasks_per_minute: int = field(
        default_factory=partial(_read_env_int, name="MAX_TASKS_PER_MINUTE", default=60)
    )
    num_retries: int = field(default_factory=partial(_read_env_int, name="NUM_RETRIES", default=3))
    timeout_seconds: int = field(
        default_factory=partial(_read_env_int, name="TIMEOUT_SECONDS", default=30)
    )
    use_session_pool: bool = field(
        default_factory=partial(_read_env_bool, name="USE_SESSION_POOL", default=True)
    )
    retry_on_blocked: bool = field(
        default_factory=partial(_read_env_bool, name="RETRY_ON_BLOCKED", default=True)
    )
    max_session_pool_size: int = field(
        default_factory=partial(_read_env_int, name="MAX_SESSION_POOL_SIZE", default=10)
    )
    max_session_rotations: int = field(
        default_factory=partial(_read_env_int, name="MAX_SESSION_ROTATIONS", default=5)
    )
    # Goodreads returns 429 (rate-limited) under heavy concurrency; treating
    # it as a blocked-response trigger lets Crawlee rotate sessions instead
    # of burning through retries on a session that's already flagged.
    additional_http_error_status_codes: tuple[int, ...] = (429,)

    label_book: str = "BOOK"
    label_author: str = "AUTHOR"
    html_parser: str = "lxml"

    @property
    def base_book_url(self) -> str:
        """Base URL prefix a book ID is appended to.

        Returns:
            The complete base URL string (e.g. 'https://www.goodreads.com/book/show/').
        """
        return f"{self.root_url}/book/show/"


SETTINGS = Settings()
