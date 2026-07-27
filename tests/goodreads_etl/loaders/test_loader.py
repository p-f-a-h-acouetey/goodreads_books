"""Pytest suite for src.goodreads_etl.loaders.load."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest
from huggingface_hub.utils import EntryNotFoundError

from src.goodreads_etl.loaders.load import (
    _extract_part_numbers,
    _load_hub_scraped_ids,
    _resolve_hf_token,
    append_local_scraped_ids,
    ensure_repo_exists,
    get_hf_api,
    get_next_part_number,
    load_local_scraped_ids,
    push_checkpoint_to_hub,
    save_book_ids_to_hub,
)
from src.goodreads_etl.utils.config_setter import SETTINGS, Settings


def make_settings(**overrides: Any) -> Settings:
    """Factory helper to construct a dummy `Settings` instance.

    Args:
        **overrides: Optional attribute overrides for default settings.

    Returns:
        Settings: A updated Settings instance with applied overrides.
    """
    return replace(SETTINGS, **overrides)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestResolveHfToken:
    """Tests evaluating Hugging Face API authentication token resolution."""

    def test_returns_explicit_token_when_provided(self) -> None:
        """Verify an explicitly passed token takes absolute priority."""
        assert _resolve_hf_token(explicit_token="explicit") == "explicit"

    def test_falls_back_to_env_var_when_no_explicit_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify fallback to HF_TOKEN environment variable when no explicit token is given."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "from-env")
        assert _resolve_hf_token(explicit_token=None) == "from-env"

    def test_returns_none_when_neither_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify None is returned when neither explicit token nor environment variable exists."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert _resolve_hf_token(explicit_token=None) is None


class TestGetHfApi:
    """Tests evaluating HfApi client initialization and token binding."""

    def test_builds_api_client_with_resolved_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify HfApi client is instantiated with the environment token when unsupplied."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "env-token")
        api = get_hf_api()
        assert api.token == "env-token"

    def test_explicit_token_takes_precedence_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify explicit token parameter overrides environment variable during client creation."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "env-token")
        api = get_hf_api(hf_token="explicit-token")
        assert api.token == "explicit-token"


# ---------------------------------------------------------------------------
# Repo setup
# ---------------------------------------------------------------------------


class TestEnsureRepoExists:
    """Tests evaluating Hugging Face repository existence check and creation."""

    def test_calls_create_repo_with_expected_arguments(self) -> None:
        """Verify dataset repository creation is invoked with correct specified arguments."""
        api = MagicMock()
        ensure_repo_exists(api=api, repo_id="org/dataset")

        api.create_repo.assert_called_once_with(
            repo_id="org/dataset", repo_type="dataset", private=False, exist_ok=True
        )

    def test_uses_default_repo_id_when_not_provided(self) -> None:
        """Verify default repository ID from settings is used when omitted."""
        api = MagicMock()
        ensure_repo_exists(api=api)

        api.create_repo.assert_called_once_with(
            repo_id=SETTINGS.repo_id, repo_type="dataset", private=False, exist_ok=True
        )


# ---------------------------------------------------------------------------
# Checkpoint numbering
# ---------------------------------------------------------------------------


class TestExtractPartNumbers:
    """Tests evaluating regex parsing of checkpoint file part numbers."""

    def test_extracts_matching_part_numbers(self) -> None:
        """Verify part integers are correctly parsed from filenames matching template."""
        files = ["books-part1.parquet", "books-part2.parquet", "other.txt"]
        result = _extract_part_numbers(
            files=files, filename_template="books-part{part}.parquet"
        )
        assert sorted(result) == [1, 2]

    def test_returns_empty_list_when_no_files_match(self) -> None:
        """Verify empty list is returned when filenames do not match expected template."""
        files = ["random.txt", "notes.md"]
        result = _extract_part_numbers(
            files=files, filename_template="books-part{part}.parquet"
        )
        assert result == []

    def test_returns_empty_list_for_empty_file_list(self) -> None:
        """Verify empty list is returned when provided file sequence is empty."""
        result = _extract_part_numbers(
            files=[], filename_template="books-part{part}.parquet"
        )
        assert result == []


class TestGetNextPartNumber:
    """Tests evaluating target part sequence number determination."""

    def test_returns_one_when_no_files_exist(self) -> None:
        """Verify sequence defaults to part 1 when no existing files are found in repo."""
        api = MagicMock()
        api.list_repo_files.return_value = []
        settings = make_settings()

        assert get_next_part_number(api=api, settings=settings) == 1

    def test_returns_max_plus_one_when_files_exist(self) -> None:
        """Verify next part number is calculated as maximum existing part number plus one."""
        api = MagicMock()
        api.list_repo_files.return_value = [
            "books-part1.parquet",
            "books-part9.parquet",
        ]
        settings = make_settings()

        assert get_next_part_number(api=api, settings=settings) == 10

    def test_defaults_to_one_when_listing_raises(self, mocker: pytest.MonkeyPatch) -> None:
        """Verify failure during repo listing logs a warning and gracefully defaults to part 1."""
        warning_mock = mocker.patch("src.goodreads_etl.loaders.load.logger.warning")
        api = MagicMock()
        api.list_repo_files.side_effect = Exception("network error")
        settings = make_settings()

        assert get_next_part_number(api=api, settings=settings) == 1
        warning_mock.assert_called_once()


# ---------------------------------------------------------------------------
# push_checkpoint_to_hub
# ---------------------------------------------------------------------------


class TestPushCheckpointToHub:
    """Tests evaluating Parquet checkpoint uploads to Hugging Face Hub."""

    def test_returns_none_without_uploading_when_dataframe_empty(self) -> None:
        """Verify early exit without API upload attempt when DataFrame contains no rows."""
        api = MagicMock()
        dataframe = pl.DataFrame({"book_id": []})

        result = push_checkpoint_to_hub(
            dataframe=dataframe, part_number=1, api=api, settings=make_settings()
        )

        assert result is None
        api.upload_file.assert_not_called()

    def test_uploads_parquet_file_with_expected_metadata(self) -> None:
        """Verify Parquet file upload is triggered with correct destination path and commit metadata."""
        api = MagicMock()
        dataframe = pl.DataFrame({"book_id": ["1", "2"]})
        settings = make_settings()

        push_checkpoint_to_hub(
            dataframe=dataframe, part_number=3, api=api, settings=settings
        )

        api.upload_file.assert_called_once()
        call_kwargs = api.upload_file.call_args.kwargs
        assert call_kwargs["path_in_repo"] == "books-part3.parquet"
        assert call_kwargs["repo_id"] == settings.repo_id
        assert call_kwargs["repo_type"] == "dataset"
        assert "2 books" in call_kwargs["commit_message"]

    def test_logs_info_after_successful_push(self, mocker: pytest.MonkeyPatch) -> None:
        """Verify informational message is logged upon completing successful file upload."""
        info_mock = mocker.patch("src.goodreads_etl.loaders.load.logger.info")
        api = MagicMock()
        dataframe = pl.DataFrame({"book_id": ["1"]})

        push_checkpoint_to_hub(
            dataframe=dataframe, part_number=1, api=api, settings=make_settings()
        )

        info_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Local scraped-ID tracker
# ---------------------------------------------------------------------------


class TestLoadLocalScrapedIds:
    """Tests evaluating reading scraped book IDs from local filesystem storage."""

    def test_returns_empty_set_when_file_does_not_exist(self, tmp_path: Path) -> None:
        """Verify an empty set is returned when the target local ID file does not exist."""
        settings = make_settings(local_scraped_ids_path=str(tmp_path / "missing.txt"))
        assert load_local_scraped_ids(settings=settings) == set()

    def test_reads_ids_from_existing_file(self, tmp_path: Path) -> None:
        """Verify book IDs are properly parsed into a set from existing file."""
        file_path = tmp_path / "ids.txt"
        file_path.write_text("1\n2\n3\n", encoding="utf-8")
        settings = make_settings(local_scraped_ids_path=str(file_path))

        assert load_local_scraped_ids(settings=settings) == {"1", "2", "3"}

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        """Verify empty lines and superfluous whitespace are filtered out during loading."""
        file_path = tmp_path / "ids.txt"
        file_path.write_text("1\n\n\n2\n", encoding="utf-8")
        settings = make_settings(local_scraped_ids_path=str(file_path))

        assert load_local_scraped_ids(settings=settings) == {"1", "2"}


class TestAppendLocalScrapedIds:
    """Tests evaluating writing and appending scraped book IDs to local storage."""

    def test_returns_none_and_writes_nothing_for_empty_list(
        self, tmp_path: Path
    ) -> None:
        """Verify early return with no disk write operations when ID list is empty."""
        file_path = tmp_path / "ids.txt"
        settings = make_settings(local_scraped_ids_path=str(file_path))

        result = append_local_scraped_ids(book_ids=[], settings=settings)

        assert result is None
        assert not file_path.exists()

    def test_creates_parent_directories_and_writes_ids(
        self, tmp_path: Path
    ) -> None:
        """Verify missing parent directories are created automatically before writing IDs."""
        file_path = tmp_path / "nested" / "ids.txt"
        settings = make_settings(local_scraped_ids_path=str(file_path))

        append_local_scraped_ids(book_ids=["1", "2"], settings=settings)

        assert file_path.read_text(encoding="utf-8") == "1\n2\n"

    def test_appends_to_existing_file_without_overwriting(
        self, tmp_path: Path
    ) -> None:
        """Verify new IDs are appended to existing file content without overwriting."""
        file_path = tmp_path / "ids.txt"
        file_path.write_text("1\n", encoding="utf-8")
        settings = make_settings(local_scraped_ids_path=str(file_path))

        append_local_scraped_ids(book_ids=["2"], settings=settings)

        assert file_path.read_text(encoding="utf-8") == "1\n2\n"

    def test_logs_info_after_successful_append(
        self, tmp_path: Path, mocker: pytest.MonkeyPatch
    ) -> None:
        """Verify informational message is logged following successful file write."""
        info_mock = mocker.patch("src.goodreads_etl.loaders.load.logger.info")
        file_path = tmp_path / "ids.txt"
        settings = make_settings(local_scraped_ids_path=str(file_path))

        append_local_scraped_ids(book_ids=["1"], settings=settings)

        info_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Hub-hosted scraped-ID tracker
# ---------------------------------------------------------------------------


class TestLoadHubScrapedIds:
    """Tests evaluating retrieval of remote scraped ID records from Hugging Face Hub."""

    def test_returns_empty_set_when_file_not_found_on_hub(
        self, mocker: pytest.MonkeyPatch
    ) -> None:
        """Verify empty set is returned when remote ID tracker file is missing on Hub."""
        mocker.patch(
            "src.goodreads_etl.loaders.load.hf_hub_download",
            side_effect=EntryNotFoundError("missing"),
        )
        api = MagicMock(token="tok")

        assert _load_hub_scraped_ids(api=api, settings=make_settings()) == set()

    def test_returns_empty_set_and_logs_warning_on_other_errors(
        self, mocker: pytest.MonkeyPatch
    ) -> None:
        """Verify generic download errors log a warning and return an empty set."""
        warning_mock = mocker.patch("src.goodreads_etl.loaders.load.logger.warning")
        mocker.patch(
            "src.goodreads_etl.loaders.load.hf_hub_download",
            side_effect=Exception("network error"),
        )
        api = MagicMock(token="tok")

        assert _load_hub_scraped_ids(api=api, settings=make_settings()) == set()
        warning_mock.assert_called_once()

    def test_reads_ids_from_downloaded_file(
        self, mocker: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify downloaded remote ID tracker file is correctly parsed into a set."""
        file_path = tmp_path / "downloaded.txt"
        file_path.write_text("1\n2\n", encoding="utf-8")
        mocker.patch(
            "src.goodreads_etl.loaders.load.hf_hub_download",
            return_value=str(file_path),
        )
        api = MagicMock(token="tok")

        assert _load_hub_scraped_ids(api=api, settings=make_settings()) == {"1", "2"}


class TestSaveBookIdsToHub:
    """Tests evaluating synchronization and upload of scraped IDs to Hugging Face Hub."""

    def test_returns_none_without_uploading_for_empty_list(
        self, mocker: pytest.MonkeyPatch
    ) -> None:
        """Verify early return without upload when input book ID list is empty."""
        mocker.patch(
            "src.goodreads_etl.loaders.load._load_hub_scraped_ids",
            return_value=set(),
        )
        api = MagicMock()

        result = save_book_ids_to_hub(
            book_ids=[], api=api, settings=make_settings()
        )

        assert result is None
        api.upload_file.assert_not_called()

    def test_merges_new_ids_with_existing_and_uploads(
        self, mocker: pytest.MonkeyPatch
    ) -> None:
        """Verify new IDs are merged with existing remote records prior to Hub upload."""
        mocker.patch(
            "src.goodreads_etl.loaders.load._load_hub_scraped_ids",
            return_value={"1"},
        )
        api = MagicMock()
        settings = make_settings()

        save_book_ids_to_hub(book_ids=["2"], api=api, settings=settings)

        api.upload_file.assert_called_once()
        call_kwargs = api.upload_file.call_args.kwargs
        assert call_kwargs["path_in_repo"] == settings.book_ids_filename
        assert call_kwargs["repo_id"] == settings.repo_id
        assert "1 new IDs" in call_kwargs["commit_message"]

    def test_deduplicates_ids_preserving_first_occurrence_order(
        self, mocker: pytest.MonkeyPatch
    ) -> None:
        """Verify ID merge deduplicates entries while retaining original order."""
        mocker.patch(
            "src.goodreads_etl.loaders.load._load_hub_scraped_ids",
            return_value={"1"},
        )
        api = MagicMock()
        settings = make_settings()
        captured = {}

        def fake_upload_file(*, path_or_fileobj: Any, **kwargs: Any) -> None:
            captured["content"] = Path(path_or_fileobj).read_text(encoding="utf-8")

        api.upload_file.side_effect = fake_upload_file

        save_book_ids_to_hub(book_ids=["1", "2"], api=api, settings=settings)

        written_ids = captured["content"].splitlines()
        assert written_ids.count("1") == 1
        assert set(written_ids) == {"1", "2"}

    def test_logs_info_after_successful_save(
        self, mocker: pytest.MonkeyPatch
    ) -> None:
        """Verify informational message is logged following successful Hub ID upload."""
        mocker.patch(
            "src.goodreads_etl.loaders.load._load_hub_scraped_ids",
            return_value=set(),
        )
        info_mock = mocker.patch("src.goodreads_etl.loaders.load.logger.info")
        api = MagicMock()

        save_book_ids_to_hub(book_ids=["1"], api=api, settings=make_settings())

        info_mock.assert_called_once()