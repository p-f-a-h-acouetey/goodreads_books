"""Pytest suite for src.goodreads_etl.utils.config_setter."""

from __future__ import annotations

import importlib

import pytest

from src.goodreads_etl.utils import config_setter as config_module
from src.goodreads_etl.utils.config_setter import Settings, _read_env_bool, _read_env_int


@pytest.fixture(autouse=True)
def _reload_config_after_each_test():
    """Ensure every test starts and ends with a clean, unpatched module state.

    Yields:
        None: Control is passed back to the test function before resetting state.
    """
    yield
    importlib.reload(config_module)


# ---------------------------------------------------------------------------
# _read_env_int
# ---------------------------------------------------------------------------


class TestReadEnvInt:
    """Tests for the `_read_env_int` environment variable parsing helper."""

    def test_returns_default_when_env_var_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify fallback to default integer value when environment variable is missing."""
        monkeypatch.delenv("SOME_INT_VAR", raising=False)
        assert _read_env_int(name="SOME_INT_VAR", default=42) == 42

    def test_returns_parsed_int_when_env_var_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify successful integer parsing when valid numeric string is present."""
        monkeypatch.delenv("SOME_INT_VAR", raising=False)
        monkeypatch.setenv("SOME_INT_VAR", "123")
        assert _read_env_int(name="SOME_INT_VAR", default=42) == 123

    def test_raises_value_error_when_env_var_not_numeric(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify ValueError is raised when environment variable cannot be cast to int."""
        monkeypatch.delenv("SOME_INT_VAR", raising=False)
        monkeypatch.setenv("SOME_INT_VAR", "not-a-number")
        with pytest.raises(ValueError):
            _read_env_int(name="SOME_INT_VAR", default=42)

    def test_empty_string_env_var_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify ValueError is raised when environment variable is an empty string."""
        monkeypatch.delenv("SOME_INT_VAR", raising=False)
        monkeypatch.setenv("SOME_INT_VAR", "")
        with pytest.raises(ValueError):
            _read_env_int(name="SOME_INT_VAR", default=42)


# ---------------------------------------------------------------------------
# _read_env_bool
# ---------------------------------------------------------------------------


class TestReadEnvBool:
    """Tests for the `_read_env_bool` environment variable parsing helper."""

    def test_returns_default_when_env_var_unset_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify fallback to True default when environment variable is missing."""
        monkeypatch.delenv("SOME_BOOL_VAR", raising=False)
        assert _read_env_bool(name="SOME_BOOL_VAR", default=True) is True

    def test_returns_default_when_env_var_unset_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify fallback to False default when environment variable is missing."""
        monkeypatch.delenv("SOME_BOOL_VAR", raising=False)
        assert _read_env_bool(name="SOME_BOOL_VAR", default=False) is False

    @pytest.mark.parametrize("value", ["1", "true ", "True", "TRUE", "yes", " YES ", "on", "On"])
    def test_returns_true_for_truthy_strings(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Verify string representations of truthy values parse correctly to True.

        Args:
            monkeypatch: Pytest fixture for patching environment variables.
            value: Parameterized truthy string representation.
        """
        monkeypatch.setenv("SOME_BOOL_VAR", value)
        assert _read_env_bool(name="SOME_BOOL_VAR", default=False) is True

    @pytest.mark.parametrize("value", ["0", "false      ", "False", "no", "off", "ran-xdom", ""])
    def test_returns_false_for_non_truthy_strings(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Verify falsy or unrecognized strings parse correctly to False.

        Args:
            monkeypatch: Pytest fixture for patching environment variables.
            value: Parameterized non-truthy string representation.
        """
        monkeypatch.setenv("SOME_BOOL_VAR", value)
        assert _read_env_bool(name="SOME_BOOL_VAR", default=True) is False


# ---------------------------------------------------------------------------
# Settings: defaults (no env vars set)
# ---------------------------------------------------------------------------


class TestSettingsDefaults:
    """Tests default values for `Settings` when no environment overrides are defined."""

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch):
        """Clear all relevant environment variables before running each default test.

        Args:
            monkeypatch: Pytest fixture for modifying environment context.

        Yields:
            None: Context passed to individual default value test methods.
        """
        env_vars = [
            "GOODREADS_ROOT_URL",
            "HF_REPO_ID",
            "LOCAL_SCRAPED_IDS_PATH",
            "SAMPLE_SIZE",
            "CHECKPOINT_EVERY",
            "MIN_BOOK_ID",
            "MAX_BOOK_ID",
            "MAX_RANDOM_ID_ATTEMPTS",
            "MAX_CONCURRENCY",
            "MAX_TASKS_PER_MINUTE",
            "NUM_RETRIES",
            "TIMEOUT_SECONDS",
            "USE_SESSION_POOL",
            "RETRY_ON_BLOCKED",
            "MAX_SESSION_POOL_SIZE",
            "MAX_SESSION_ROTATIONS",
        ]
        for var in env_vars:
            monkeypatch.delenv(var, raising=False)
        yield

    def test_default_root_url(self) -> None:
        """Verify default root URL for Goodreads target."""
        assert Settings().root_url == "https://www.goodreads.com"

    def test_default_repo_id(self) -> None:
        """Verify default Hugging Face repository target ID."""
        assert Settings().repo_id == "pfaha/goodreads-books"

    def test_default_encoding(self) -> None:
        """Verify default string encoding standard."""
        assert Settings().encoding == "utf-8"

    def test_default_book_ids_filename(self) -> None:
        """Verify default output filename for scraped book ID list."""
        assert Settings().book_ids_filename == "scraped_book_ids.txt"

    def test_default_checkpoint_filename_template(self) -> None:
        """Verify default template string for parquet checkpoint filenames."""
        assert Settings().checkpoint_filename_template == "books-part{part}.parquet"

    def test_default_local_scraped_ids_path(self) -> None:
        """Verify default local file system path for ID tracking."""
        assert Settings().local_scraped_ids_path == "data/scraped_book_ids.txt"

    def test_default_sample_size(self) -> None:
        """Verify default sampling target count."""
        assert Settings().sample_size == 100_000

    def test_default_checkpoint_every(self) -> None:
        """Verify default batch size threshold for pushing checkpoints."""
        assert Settings().checkpoint_every == 1000

    def test_default_min_book_id(self) -> None:
        """Verify default minimum book ID boundary."""
        assert Settings().min_book_id == 1

    def test_default_max_book_id(self) -> None:
        """Verify default maximum book ID boundary."""
        assert Settings().max_book_id == 60_000_000

    def test_default_max_random_id_attempts(self) -> None:
        """Verify default upper bound on random ID generation attempts."""
        assert Settings().max_random_id_attempts == 10_000

    def test_default_max_concurrency(self) -> None:
        """Verify default asynchronous concurrency limit."""
        assert Settings().max_concurrency == 50

    def test_default_max_tasks_per_minute(self) -> None:
        """Verify default rate limiter constraint per minute."""
        assert Settings().max_tasks_per_minute == 60

    def test_default_num_retries(self) -> None:
        """Verify default HTTP retry count limit."""
        assert Settings().num_retries == 3

    def test_default_timeout_seconds(self) -> None:
        """Verify default request timeout setting in seconds."""
        assert Settings().timeout_seconds == 30

    def test_default_use_session_pool(self) -> None:
        """Verify HTTP session pooling is enabled by default."""
        assert Settings().use_session_pool is True

    def test_default_retry_on_blocked(self) -> None:
        """Verify automatic retries on blocked/throttled status codes is enabled by default."""
        assert Settings().retry_on_blocked is True

    def test_default_max_session_pool_size(self) -> None:
        """Verify default maximum capacity for session pool."""
        assert Settings().max_session_pool_size == 10

    def test_default_max_session_rotations(self) -> None:
        """Verify default session rotation count threshold."""
        assert Settings().max_session_rotations == 5

    def test_default_additional_http_error_status_codes(self) -> None:
        """Verify default status code tuple treated as error conditions."""
        assert Settings().additional_http_error_status_codes == (429,)

    def test_default_label_book(self) -> None:
        """Verify default label identifying book record entities."""
        assert Settings().label_book == "BOOK"

    def test_default_label_author(self) -> None:
        """Verify default label identifying author record entities."""
        assert Settings().label_author == "AUTHOR"

    def test_default_html_parser(self) -> None:
        """Verify default HTML parser engine configured for BeautifulSoup."""
        assert Settings().html_parser == "lxml"

    def test_default_base_book_url_property(self) -> None:
        """Verify computed base URL property for individual book detail pages."""
        assert Settings().base_book_url == "https://www.goodreads.com/book/show/"
