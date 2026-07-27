"""Pytest suite for src.goodreads_etl.pipelines.run_pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Set
from unittest.mock import MagicMock

import pytest

from src.goodreads_etl.pipelines.run_pipeline import (
    _load_resume_state,
    _persist_batch,
    _run_one_batch,
    scrape_books,
)
from src.goodreads_etl.utils.book_recorder import BookRecord
from src.goodreads_etl.utils.config_setter import SETTINGS, Settings

MODULE = "src.goodreads_etl.pipelines.run_pipeline"


def make_settings(**overrides: Any) -> Settings:
    """Factory helper to construct a dummy `Settings` instance.

    Args:
        **overrides: Optional attribute overrides for default settings.

    Returns:
        Settings: An updated Settings instance with applied overrides.
    """
    return replace(SETTINGS, **overrides)


def make_record(**overrides: Any) -> BookRecord:
    """Factory helper to construct a dummy `BookRecord` instance.

    Args:
        **overrides: Optional field overrides for default book record attributes.

    Returns:
        BookRecord: A sample BookRecord instance populated with test data.
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


@pytest.fixture
def api() -> MagicMock:
    """Fixture providing a mocked Hugging Face API client instance.

    Returns:
        MagicMock: Mocked API object.
    """
    return MagicMock()


@pytest.fixture
def scrape_mocks(mocker: pytest.MonkeyPatch, api: MagicMock) -> Dict[str, MagicMock]:
    """Patch common scrape_books dependencies and return handles for assertions.

    Args:
        mocker: Pytest monkeypatch/mocker fixture.
        api: Mocked Hugging Face API client fixture.

    Returns:
        Dict[str, MagicMock]: A dictionary mapping dependency names to their respective mocks.
    """
    mocker.patch(f"{MODULE}.get_hf_api", return_value=api)
    return {
        "ensure_repo_exists": mocker.patch(f"{MODULE}.ensure_repo_exists"),
        "load_resume_state": mocker.patch(
            f"{MODULE}._load_resume_state", return_value=(set(), [])
        ),
        "get_next_part_number": mocker.patch(
            f"{MODULE}.get_next_part_number", return_value=1
        ),
        "save_book_ids_to_hub": mocker.patch(f"{MODULE}.save_book_ids_to_hub"),
    }


# ---------------------------------------------------------------------------
# _load_resume_state
# ---------------------------------------------------------------------------


class TestLoadResumeState:
    """Tests evaluating resume state initialization and tracker loading."""

    def test_returns_empty_state_when_resume_is_false(
        self, mocker: pytest.MonkeyPatch
    ) -> None:
        """Verify an empty set and list are returned without disk reads when resume is disabled.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
        """
        load_mock = mocker.patch(f"{MODULE}.load_local_scraped_ids")

        tried, sampled = _load_resume_state(resume=False, settings=make_settings())

        assert (tried, sampled) == (set(), [])
        load_mock.assert_not_called()

    @pytest.mark.parametrize("stored_ids", [{"1", "2"}, {"1"}, set()])
    def test_loads_ids_from_local_tracker_when_resume_is_true(
        self, mocker: pytest.MonkeyPatch, stored_ids: set[str]
    ) -> None:
        """Verify stored IDs are loaded from local tracker when resume is enabled.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
            stored_ids: Parameterized set of pre-existing scraped IDs.
        """
        mocker.patch(f"{MODULE}.load_local_scraped_ids", return_value=stored_ids)

        tried, sampled = _load_resume_state(resume=True, settings=make_settings())

        assert tried == stored_ids
        assert set(sampled) == stored_ids

    def test_returned_tried_set_and_sampled_list_are_independent_copies(
        self, mocker: pytest.MonkeyPatch
    ) -> None:
        """Verify returned tried set and sampled list are independent copies to prevent state mutation leaks.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
        """
        mocker.patch(f"{MODULE}.load_local_scraped_ids", return_value={"1"})

        tried, sampled = _load_resume_state(resume=True, settings=make_settings())
        tried.add("2")
        sampled.append("3")

        assert tried == {"1", "2"}
        assert sampled == ["1", "3"]


# ---------------------------------------------------------------------------
# _run_one_batch
# ---------------------------------------------------------------------------


class TestRunOneBatch:
    """Tests evaluating asynchronous single-batch sampling execution."""

    @pytest.mark.asyncio
    async def test_delegates_to_run_sampling_crawl_with_expected_arguments(
        self, mocker: pytest.MonkeyPatch
    ) -> None:
        """Verify sampling crawl delegate is invoked with accurate parameters and settings.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
        """
        crawl_mock = mocker.patch(
            f"{MODULE}.run_sampling_crawl", return_value=[make_record()]
        )
        settings = make_settings(min_book_id=5, max_book_id=50)

        result = await _run_one_batch(
            batch_size=10, already_tried_book_ids={"1"}, settings=settings
        )

        assert result == [make_record()]
        crawl_mock.assert_awaited_once_with(
            sample_size=10,
            min_book_id=5,
            max_book_id=50,
            settings=settings,
            already_tried_book_ids={"1"},
        )


# ---------------------------------------------------------------------------
# _persist_batch
# ---------------------------------------------------------------------------


class TestPersistBatch:
    """Tests evaluating batch persistence across remote Hub and local storage."""

    @pytest.fixture(autouse=True)
    def _patch_io(self, mocker: pytest.MonkeyPatch) -> None:
        """Autouse fixture setting up standard IO patches for push and append methods.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
        """
        self.push_mock = mocker.patch(f"{MODULE}.push_checkpoint_to_hub")
        self.append_mock = mocker.patch(f"{MODULE}.append_local_scraped_ids")

    def test_returns_book_ids_and_wires_push_and_append_calls(
        self, mocker: pytest.MonkeyPatch, api: MagicMock
    ) -> None:
        """Verify extracted book IDs are returned and IO handlers receive correct arguments.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
            api: Mocked Hugging Face API client fixture.
        """
        records = [make_record(book_id="1"), make_record(book_id="2")]
        settings = make_settings()

        result = _persist_batch(
            records=records, part_number=7, api=api, settings=settings
        )

        assert result == ["1", "2"]

        push_kwargs = self.push_mock.call_args.kwargs
        assert push_kwargs["part_number"] == 7
        assert push_kwargs["api"] is api
        assert push_kwargs["settings"] is settings
        assert push_kwargs["dataframe"].height == 2

        self.append_mock.assert_called_once_with(
            book_ids=["1", "2"], settings=settings
        )


# ---------------------------------------------------------------------------
# scrape_books (top-level orchestration)
# ---------------------------------------------------------------------------


class TestScrapeBooks:
    """Tests evaluating top-level pipeline orchestration and lifecycle management."""

    def test_ensures_repo_exists_before_anything_else(
        self,
        mocker: pytest.MonkeyPatch,
        scrape_mocks: Dict[str, MagicMock],
        api: MagicMock,
    ) -> None:
        """Verify target repository existence is validated before initiating pipeline execution.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
            scrape_mocks: Dictionary of patched pipeline dependencies.
            api: Mocked Hugging Face API client fixture.
        """
        settings = make_settings(sample_size=0)

        scrape_books(settings=settings)

        scrape_mocks["ensure_repo_exists"].assert_called_once_with(
            api=api, repo_id=settings.repo_id
        )

    def test_skips_crawl_loop_when_resume_already_meets_target(
        self,
        mocker: pytest.MonkeyPatch,
        scrape_mocks: Dict[str, MagicMock],
        api: MagicMock,
    ) -> None:
        """Verify crawl loop is skipped when resumed state already satisfies sample size target.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
            scrape_mocks: Dictionary of patched pipeline dependencies.
            api: Mocked Hugging Face API client fixture.
        """
        scrape_mocks["load_resume_state"].return_value = ({"1", "2"}, ["1", "2"])
        run_batch_mock = mocker.patch(f"{MODULE}._run_one_batch")
        settings = make_settings(sample_size=2)

        scrape_books(settings=settings)

        run_batch_mock.assert_not_called()
        scrape_mocks["save_book_ids_to_hub"].assert_called_once_with(
            book_ids=["1", "2"], api=api, settings=settings
        )

    def test_runs_single_batch_when_target_fits_in_one_checkpoint(
        self,
        mocker: pytest.MonkeyPatch,
        scrape_mocks: Dict[str, MagicMock],
        api: MagicMock,
    ) -> None:
        """Verify single batch run completes pipeline when sample target fits within checkpoint limit.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
            scrape_mocks: Dictionary of patched pipeline dependencies.
            api: Mocked Hugging Face API client fixture.
        """

        async def fake_run_one_batch(
            *, batch_size: int, already_tried_book_ids: Set[str], settings: Settings
        ) -> list[BookRecord]:
            return [make_record(book_id="1"), make_record(book_id="2")]

        mocker.patch(f"{MODULE}._run_one_batch", side_effect=fake_run_one_batch)
        persist_mock = mocker.patch(
            f"{MODULE}._persist_batch", return_value=["1", "2"]
        )
        settings = make_settings(sample_size=2, checkpoint_every=1000)

        scrape_books(settings=settings)

        persist_mock.assert_called_once()
        scrape_mocks["save_book_ids_to_hub"].assert_called_once_with(
            book_ids=["1", "2"], api=api, settings=settings
        )

    def test_runs_multiple_batches_until_target_reached(
        self,
        mocker: pytest.MonkeyPatch,
        scrape_mocks: Dict[str, MagicMock],
        api: MagicMock,
    ) -> None:
        """Verify pipeline executes sequential batches with incrementing part numbers until target is met.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
            scrape_mocks: Dictionary of patched pipeline dependencies.
            api: Mocked Hugging Face API client fixture.
        """

        async def fake_run_one_batch(
            *, batch_size: int, already_tried_book_ids: Set[str], settings: Settings
        ) -> list[BookRecord]:
            book_id = "1" if not already_tried_book_ids else "2"
            return [make_record(book_id=book_id)]

        mocker.patch(f"{MODULE}._run_one_batch", side_effect=fake_run_one_batch)
        persist_mock = mocker.patch(
            f"{MODULE}._persist_batch", side_effect=[["1"], ["2"]]
        )
        settings = make_settings(sample_size=2, checkpoint_every=1)

        scrape_books(settings=settings)

        assert persist_mock.call_count == 2
        parts = [call.kwargs["part_number"] for call in persist_mock.call_args_list]
        assert parts == [1, 2]
        scrape_mocks["save_book_ids_to_hub"].assert_called_once_with(
            book_ids=["1", "2"], api=api, settings=settings
        )

    def test_stops_early_and_logs_error_when_batch_returns_no_records(
        self,
        mocker: pytest.MonkeyPatch,
        scrape_mocks: Dict[str, MagicMock],
        api: MagicMock,
    ) -> None:
        """Verify pipeline terminates early and logs an error when a batch yields no new records.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
            scrape_mocks: Dictionary of patched pipeline dependencies.
            api: Mocked Hugging Face API client fixture.
        """

        async def fake_run_one_batch(
            *, batch_size: int, already_tried_book_ids: Set[str], settings: Settings
        ) -> list[BookRecord]:
            return []

        mocker.patch(f"{MODULE}._run_one_batch", side_effect=fake_run_one_batch)
        persist_mock = mocker.patch(f"{MODULE}._persist_batch")
        error_mock = mocker.patch(f"{MODULE}.logger.error")
        settings = make_settings(sample_size=5, checkpoint_every=5)

        scrape_books(settings=settings)

        persist_mock.assert_not_called()
        error_mock.assert_called_once()
        scrape_mocks["save_book_ids_to_hub"].assert_called_once_with(
            book_ids=[], api=api, settings=settings
        )

    def test_batch_size_capped_by_checkpoint_every(
        self,
        mocker: pytest.MonkeyPatch,
        scrape_mocks: Dict[str, MagicMock],
        api: MagicMock,
    ) -> None:
        """Verify batch size is correctly capped by the checkpoint_every configuration limit.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
            scrape_mocks: Dictionary of patched pipeline dependencies.
            api: Mocked Hugging Face API client fixture.
        """
        captured_batch_sizes = []

        async def fake_run_one_batch(
            *, batch_size: int, already_tried_book_ids: Set[str], settings: Settings
        ) -> list[BookRecord]:
            captured_batch_sizes.append(batch_size)
            return [make_record(book_id=str(len(captured_batch_sizes)))]

        mocker.patch(f"{MODULE}._run_one_batch", side_effect=fake_run_one_batch)
        mocker.patch(
            f"{MODULE}._persist_batch",
            side_effect=lambda **kwargs: [r.book_id for r in kwargs["records"]],
        )
        settings = make_settings(sample_size=5, checkpoint_every=2)

        scrape_books(settings=settings)

        assert captured_batch_sizes[0] == 2

    def test_already_tried_book_ids_grow_across_batches(
        self,
        mocker: pytest.MonkeyPatch,
        scrape_mocks: Dict[str, MagicMock],
        api: MagicMock,
    ) -> None:
        """Verify set of tried book IDs accumulates continuously across successive batches.

        Args:
            mocker: Pytest monkeypatch/mocker fixture.
            scrape_mocks: Dictionary of patched pipeline dependencies.
            api: Mocked Hugging Face API client fixture.
        """
        captured_tried_ids = []

        async def fake_run_one_batch(
            *, batch_size: int, already_tried_book_ids: Set[str], settings: Settings
        ) -> list[BookRecord]:
            captured_tried_ids.append(set(already_tried_book_ids))
            book_id = str(len(captured_tried_ids))
            return [make_record(book_id=book_id)]

        mocker.patch(f"{MODULE}._run_one_batch", side_effect=fake_run_one_batch)
        mocker.patch(
            f"{MODULE}._persist_batch",
            side_effect=lambda **kwargs: [r.book_id for r in kwargs["records"]],
        )
        settings = make_settings(sample_size=2, checkpoint_every=1)

        scrape_books(settings=settings)

        assert captured_tried_ids == [set(), {"1"}]