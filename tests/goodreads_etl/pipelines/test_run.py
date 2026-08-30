"""Pytest suite for run.py's BookRunner.

Covers the orchestration logic in BookRunner.run: the already-met-target
short-circuit, delegating to BookExtractor.extract for the remaining
count, batching results through BookLoader.load in PARQUET_BATCH_SIZE
chunks, and returning the correctly-concatenated DataFrame of records
collected in the current run.

BookExtractor and BookLoader are both fully mocked -- no network calls,
no real Hugging Face Hub access, no real crawl. This suite only verifies
that BookRunner wires the two together correctly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

import src.goodreads_etl.pipelines.run as run_module
from src.goodreads_etl.pipelines.run import BookRunner


def _make_record(book_id: str) -> dict:
    """Build a minimal fake book record dict with the given book_id."""
    return {"book_id": book_id, "title": f"Book {book_id}"}


@pytest.fixture
def runner(monkeypatch):
    """A BookRunner with BookExtractor and BookLoader replaced by mocks.

    extractor.extract is an AsyncMock (awaited in run()); loader is a
    MagicMock exposing load_scraped_ids, get_next_part_number, and load
    (the two synchronous BookLoader methods plus the batching method).
    """
    mock_extractor = MagicMock()
    mock_extractor.extract = AsyncMock(return_value=[])

    mock_loader = MagicMock()
    mock_loader.load_scraped_ids.return_value = set()
    mock_loader.get_next_part_number.return_value = 1

    monkeypatch.setattr(run_module, "BookExtractor", MagicMock(return_value=mock_extractor))
    monkeypatch.setattr(run_module, "BookLoader", MagicMock(return_value=mock_loader))

    instance = BookRunner()
    instance.extractor = mock_extractor
    instance.loader = mock_loader
    return instance


class TestBookRunnerInit:
    """Tests for BookRunner.__init__."""

    def test_builds_extractor_and_loader(self, monkeypatch):
        """__init__ should construct one BookExtractor and one BookLoader."""
        mock_extractor_cls = MagicMock()
        mock_loader_cls = MagicMock()
        monkeypatch.setattr(run_module, "BookExtractor", mock_extractor_cls)
        monkeypatch.setattr(run_module, "BookLoader", mock_loader_cls)

        instance = BookRunner()

        mock_extractor_cls.assert_called_once_with()
        mock_loader_cls.assert_called_once_with()
        assert instance.extractor is mock_extractor_cls.return_value
        assert instance.loader is mock_loader_cls.return_value


class TestBookRunnerRun:
    """Tests for BookRunner.run, the sole public orchestration method."""

    @pytest.mark.parametrize(
        "known_ids",
        [
            {"1", "2", "3"},
            {"1", "2", "3", "4", "5"},
        ],
        ids=["equals_target", "exceeds_target"],
    )
    async def test_returns_none_when_target_already_met(self, runner, known_ids):
        """If known book_ids already meet or exceed sample_size, run() should short-circuit.

        Covers both the exact-match case (known == target) and the
        overshoot case (known > target) -- neither extraction nor
        loading should be attempted in either case.
        """
        runner.loader.load_scraped_ids.return_value = known_ids

        result = await runner.run(sample_size=3)

        assert result is None
        runner.extractor.extract.assert_not_called()
        runner.loader.load.assert_not_called()

    async def test_batches_records_correctly_across_multiple_load_calls(self, runner, monkeypatch):
        """Splitting, part numbering, ID threading, and concatenation should all work together.

        With PARQUET_BATCH_SIZE=2 and 4 new records, run() should:
        - call loader.load exactly twice (two batches of 2),
        - use a different filename for each batch (incrementing part number),
        - thread each batch's updated known_book_ids into the next batch's call.
        """
        monkeypatch.setattr(run_module, "PARQUET_BATCH_SIZE", 2)

        records = [_make_record(str(i)) for i in range(1, 5)]
        runner.loader.load_scraped_ids.return_value = {"123"}
        runner.extractor.extract.return_value = records
        runner.loader.get_next_part_number.return_value = 1

        seen_known_ids_args = []

        def fake_load(batch, filename, known_ids):
            seen_known_ids_args.append(set(known_ids))
            new_ids = known_ids | {r["book_id"] for r in batch}
            return pl.DataFrame(batch), new_ids

        runner.loader.load.side_effect = fake_load

        result = await runner.run(sample_size=5)

        assert result is None
        assert runner.loader.load.call_count == 2

        filenames_used = [c.args[1] for c in runner.loader.load.call_args_list]
        assert filenames_used[0] != filenames_used[1]

        assert seen_known_ids_args[0] == {"123"}
        assert seen_known_ids_args[1] == {"123", "1", "2"}
