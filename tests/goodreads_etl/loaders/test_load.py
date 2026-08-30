"""Pytest suite for load.py's BookLoader.

BookLoader.__init__ always constructs a real HfApi from the module-level
HF_TOKEN, with no dependency-injection seam. To keep BookLoader's
production code unchanged, every test in this file patches load.HF_TOKEN
and load.HfApi at the module level (via monkeypatch) BEFORE constructing
BookLoader, so __init__ picks up a fake token and a fake API class
instead of ever touching the real Hugging Face Hub. hf_hub_download is
patched the same way for load_scraped_ids. Local Parquet writes are
redirected to pytest's tmp_path fixture.
"""

from __future__ import annotations

import pytest
import polars as pl
from unittest.mock import MagicMock
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

import src.goodreads_etl.loaders.load as load_module
from src.goodreads_etl.loaders.load import HF_RAW_DIR, SCRAPED_IDS_FILENAME


# ---------------------------------------------------------------------------
# Fake HfApi -- stands in for huggingface_hub.HfApi across all tests
# ---------------------------------------------------------------------------


class FakeHfApi:
    """Records every call made to it; list_repo_files/upload_file are configurable."""

    def __init__(self, token=None):
        self.token = token
        self.list_repo_files_calls: list[dict] = []
        self.upload_file_calls: list[dict] = []
        self.list_repo_files_return: list[str] = []

    def list_repo_files(self, **kwargs):
        self.list_repo_files_calls.append(kwargs)
        return self.list_repo_files_return

    def upload_file(self, **kwargs):
        self.upload_file_calls.append(kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_output_dir(tmp_path, monkeypatch):
    """Redirect all local Parquet writes to a pytest-managed temp directory.

    Applied automatically to every test in this file so no test can ever
    write into the real project directory.
    """
    monkeypatch.setattr(load_module, "OUTPUT_DIR", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _patch_hf_token(monkeypatch):
    """Ensure HF_TOKEN is always a truthy fake value so __init__ never raises.

    Applied automatically; individual tests that need to test the missing-token
    path override this by re-patching HF_TOKEN to a falsy value themselves.
    """
    monkeypatch.setattr(load_module, "HF_TOKEN", "fake-token-for-tests")


@pytest.fixture
def loader(monkeypatch):
    """A BookLoader instance built with HfApi patched to FakeHfApi.

    Returns:
        A BookLoader whose self.api is a FakeHfApi instance, never touching
        the real Hugging Face Hub.
    """
    monkeypatch.setattr(load_module, "HfApi", FakeHfApi)
    return load_module.BookLoader()


@pytest.fixture
def fake_api(loader):
    """The FakeHfApi instance backing the loader fixture, for assertions.

    Args:
        loader: The loader fixture.

    Returns:
        The FakeHfApi instance stored as loader.api.
    """
    return loader.api


# ---------------------------------------------------------------------------
# __init__ / _get_hf_api
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for BookLoader.__init__ and the internal _get_hf_api helper."""

    @pytest.mark.parametrize("falsy_token", ["", None])
    def test_raises_when_token_missing_or_none(self, monkeypatch, falsy_token):
        """Constructing with a falsy HF_TOKEN (empty string or None) should raise RuntimeError."""
        monkeypatch.setattr(load_module, "HF_TOKEN", falsy_token)
        monkeypatch.setattr(load_module, "HfApi", FakeHfApi)
        with pytest.raises(RuntimeError, match="HF_TOKEN is not set"):
            load_module.BookLoader()

    def test_constructs_api_with_configured_token(self, monkeypatch):
        """A truthy HF_TOKEN should be passed through to the HfApi constructor."""
        monkeypatch.setattr(load_module, "HF_TOKEN", "my-real-looking-token")
        monkeypatch.setattr(load_module, "HfApi", FakeHfApi)
        instance = load_module.BookLoader()
        assert instance.api.token == "my-real-looking-token"


# ---------------------------------------------------------------------------
# get_next_part_number
# ---------------------------------------------------------------------------


class TestGetNextPartNumber:
    """Tests for BookLoader.get_next_part_number."""

    def test_returns_1_when_no_parts_exist(self, loader, fake_api):
        """An empty repo (or one with no matching files) should start numbering at 1."""
        fake_api.list_repo_files_return = []
        assert loader.get_next_part_number() == 1

    def test_returns_max_plus_one_when_parts_exist(self, loader, fake_api):
        """Only existing and matching 'books-partN.parquet' files in the correct repo should continue numbering from the max."""
        fake_api.list_repo_files_return = [
            f"{HF_RAW_DIR}/book_ids.parquet",
            f"{HF_RAW_DIR}/books-part1.parquet",
            f"{HF_RAW_DIR}/books-part2.parquet",
            f"{HF_RAW_DIR}/books-part5.parquet",
            "other_dir/books-part99.parquet",
        ]
        assert loader.get_next_part_number() == 6


# ---------------------------------------------------------------------------
# load_scraped_ids
# ---------------------------------------------------------------------------


class TestLoadScrapedIds:
    """Tests for BookLoader.load_scraped_ids."""

    def test_returns_empty_set_when_tracker_file_missing(self, loader, monkeypatch):
        """EntryNotFoundError (tracker never uploaded yet) should yield an empty set."""

        def _raise_entry_not_found(**kwargs):
            raise EntryNotFoundError("no such file")

        monkeypatch.setattr(load_module, "hf_hub_download", _raise_entry_not_found)
        assert loader.load_scraped_ids() == set()

    def test_returns_empty_set_when_repo_missing(self, loader, monkeypatch):
        """EntryNotFoundError (tracker never uploaded yet) should yield an empty set.

        Constructed without response= here since this huggingface_hub version's
        EntryNotFoundError doesn't require it (unlike RepositoryNotFoundError
        below) -- see huggingface_hub's v1.0 migration notes on the split
        between EntryNotFoundError and RemoteEntryNotFoundError.
        """

        def _raise_repo_not_found(**kwargs):
            raise RepositoryNotFoundError("no such repo", response=MagicMock())

        monkeypatch.setattr(load_module, "hf_hub_download", _raise_repo_not_found)
        assert loader.load_scraped_ids() == set()

    def test_returns_correct_ids_when_tracker_exists(self, loader, monkeypatch, tmp_path):
        """A real tracker file's book_ids should be loaded into a set correctly."""
        tracker_path = tmp_path / "book_ids.parquet"
        pl.DataFrame({"book_id": ["1", "2", "3"]}).write_parquet(tracker_path)

        monkeypatch.setattr(load_module, "hf_hub_download", lambda **kwargs: str(tracker_path))

        assert loader.load_scraped_ids() == {"1", "2", "3"}


# ---------------------------------------------------------------------------
# _save_locally
# ---------------------------------------------------------------------------


class TestSaveLocally:
    """Direct tests for the private _save_locally helper."""

    def test_save_locally(self, loader, tmp_path, monkeypatch):
        """_save_locally should create OUTPUT_DIR (and parents) if it doesn't exist yet."""
        nested_dir = tmp_path / "nested" / "output"
        monkeypatch.setattr(load_module, "OUTPUT_DIR", nested_dir)

        df = pl.DataFrame({"book_id": ["1", "2", "3"]})
        result_path = loader._save_locally(df, "book_ids.parquet")

        assert nested_dir.exists()
        assert result_path.exists()
        assert result_path == nested_dir / "book_ids.parquet"


# ---------------------------------------------------------------------------
# _upload
# ---------------------------------------------------------------------------


class TestUpload:
    """Direct tests for the private _upload helper."""

    def test_upload(self, loader, fake_api, tmp_path):
        """_upload should pass through path, repo, and commit message correctly."""
        local_path = tmp_path / "books-part1.parquet"
        local_path.write_text("dummy")

        loader._upload(local_path, "books-part1.parquet", "my commit message")

        assert len(fake_api.upload_file_calls) == 1
        call = fake_api.upload_file_calls[0]
        assert call["path_or_fileobj"] == str(local_path)
        assert call["path_in_repo"] == f"{HF_RAW_DIR}/books-part1.parquet"
        assert call["repo_id"] == load_module.HF_REPO_ID
        assert call["repo_type"] == load_module.HF_REPO_TYPE
        assert call["commit_message"] == "my commit message"


# ---------------------------------------------------------------------------
# _update_scraped_ids
# ---------------------------------------------------------------------------


class TestUpdateScrapedIds:
    """Direct tests for the private _update_scraped_ids helper."""

    def test_update_scraped_ids(self, loader, fake_api):
        """_update_scraped_ids should return the union of known and new ids."""
        result = loader._update_scraped_ids(new_book_ids=["3", "4"], known_book_ids={"1", "2"})
        tracker_calls = [c for c in fake_api.upload_file_calls if c["path_in_repo"] == f"{HF_RAW_DIR}/{SCRAPED_IDS_FILENAME}"]
        assert result == {"1", "2", "3", "4"}
        assert len(tracker_calls) == 1


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


class TestLoad:
    """Tests for BookLoader.load, the single public batch-persist entry point."""

    def test_load(self, loader, fake_api):
        """The returned DataFrame should contain exactly the records passed in."""
        records = [
            {"book_id": "3", "title": "Alpha", "average_rating": 4.5},
            {"book_id": "4", "title": "Beta", "average_rating": 3.2},
        ]
        prior_known_ids = {"1", "2"}

        df, updated_ids = loader.load(records, "books-part2.parquet", known_book_ids=prior_known_ids)
        tracker_calls = [c for c in fake_api.upload_file_calls if c["path_in_repo"] == f"{HF_RAW_DIR}/{SCRAPED_IDS_FILENAME}"]

        assert updated_ids == {"1", "2", "3", "4"}
        assert df.height == 2
        assert df.get_column("book_id").to_list() == ["3", "4"]
        assert df.get_column("title").to_list() == ["Alpha", "Beta"]
        assert df.get_column("average_rating").to_list() == [4.5, 3.2]

        assert "Update book_ids tracker (4 total)" in tracker_calls[0]["commit_message"]