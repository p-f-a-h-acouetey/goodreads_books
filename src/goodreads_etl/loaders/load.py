"""LOAD stage: batch, persist locally, and sink book records to the Hugging Face Hub.

Owns everything needed to turn a list of finished book record dicts into
checkpointed Parquet files on disk and on the Hub, plus the book_ids
tracker used across runs to avoid re-scraping the same books.

Exposes exactly one class, BookLoader, with these public methods:
- get_next_part_number()
- load_scraped_ids()
- load(records, filename, known_book_ids)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import polars as pl
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from loguru import logger

# Populates os.environ from a local .env file (e.g. HF_TOKEN) if present;
# a no-op in environments where the variables are already set externally
# (CI, Docker, etc.).
load_dotenv()

# --- Local staging output -----------------------------------------------------------
# NOTE: PARQUET_BATCH_SIZE lives in run.py, not here -- how many records
# get grouped into one checkpoint is a pipeline-orchestration concern
# (BookRunner decides chunk sizing), not a load-stage concern. BookLoader
# just persists and uploads whatever batch of records it's handed,
# regardless of size.
OUTPUT_DIR = Path(".")
PARQUET_FILENAME_TEMPLATE = "books-part{part}.parquet"
PARQUET_FILENAME_PATTERN = re.compile(r"books-part(\d+)\.parquet$")

# --- Hugging Face Hub sink -----------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = "pfaha/goodreads-books"
HF_REPO_TYPE = "dataset"
HF_RAW_DIR = "raw"

# --- Scraped-IDs tracker -----------------------------------------------------------
# Single Parquet file on the Hub listing every book_id ever successfully
# scraped, across all runs -- consulted before each run to know which IDs
# to skip and how much of the sample_size target remains.
SCRAPED_IDS_FILENAME = "book_ids.parquet"


class BookLoader:
    """Persists book batches locally, pushes them to the Hub, and tracks scraped IDs."""

    def __init__(self) -> None:
        self.logger = logger.bind(component="BookLoader")
        self.api = self._get_hf_api()

    def _get_hf_api(self) -> HfApi:
        """Build an authenticated Hugging Face Hub API client.

        Returns:
            An HfApi instance authenticated with HF_TOKEN.

        Raises:
            RuntimeError: If HF_TOKEN is not set.
        """
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN is not set -- check your .env file")
        return HfApi(token=HF_TOKEN)

    def _save_locally(self, df: pl.DataFrame, filename: str) -> Path:
        """Write a DataFrame to a local Parquet file under OUTPUT_DIR.

        Local staging happens before every Hub upload, both for the data
        batches and for the book_ids tracker, since HfApi.upload_file
        requires a real file path rather than an in-memory buffer.

        Args:
            df: The DataFrame to persist.
            filename: File name only (e.g. "books-part7.parquet").

        Returns:
            The full local path the file was written to.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        local_path = OUTPUT_DIR / filename
        df.write_parquet(local_path)
        return local_path

    def _upload(self, local_path: Path, filename: str, commit_message: str) -> None:
        """Upload a local file to {HF_REPO_ID}/{HF_RAW_DIR}/{filename}.

        Args:
            local_path: Local path of the file to upload.
            filename: File name only, used to build the path in the repo.
            commit_message: Commit message for this upload.
        """
        self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"{HF_RAW_DIR}/{filename}",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            commit_message=commit_message,
        )

    def get_next_part_number(self) -> int:
        """Determine the next books-part number to use, continuing from the Hub.

        Queries the Hub directly (rather than tracking part numbers
        locally) so that numbering stays correct even across runs on
        different machines, or after a local checkout is wiped.

        Returns:
            1 if no books-part*.parquet files exist yet, otherwise the
            highest existing part number + 1.
        """
        existing_files = self.api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
        part_numbers = [
            int(match.group(1))
            for f in existing_files
            if (match := PARQUET_FILENAME_PATTERN.search(f)) and f.startswith(f"{HF_RAW_DIR}/")
        ]
        next_part = max(part_numbers, default=0) + 1
        self.logger.info(
            f"Next part number: {next_part} ({len(part_numbers)} existing part(s) found)"
        )
        return next_part

    def load_scraped_ids(self) -> set[str]:
        """Load the set of book_ids already scraped in any previous run.

        Returns:
            Set of book_id strings already present in the tracker file, or
            an empty set if the tracker does not exist yet.
        """
        try:
            local_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                filename=f"{HF_RAW_DIR}/{SCRAPED_IDS_FILENAME}",
                token=HF_TOKEN,
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            # No tracker file yet (first-ever run) or the repo doesn't
            # exist yet -- either way, start from an empty known-IDs set.
            self.logger.info("No book_ids tracker found on the Hub yet, starting fresh")
            return set()

        scraped_ids = set(pl.read_parquet(local_path).get_column("book_id").to_list())
        self.logger.info(f"Loaded {len(scraped_ids)} already-scraped book_id(s) from the tracker")
        return scraped_ids

    def _update_scraped_ids(self, new_book_ids: list[str], known_book_ids: set[str]) -> set[str]:
        """Merge new book_ids into the tracker, save locally, and push to the Hub.

        Args:
            new_book_ids: book_ids collected in the batch just loaded.
            known_book_ids: Full set of book_ids already known before this batch.

        Returns:
            The updated, merged set of all known book_ids.
        """
        updated_ids = known_book_ids | set(new_book_ids)
        # Sorted purely for a stable, human-readable diff between commits
        # on the Hub -- set order isn't otherwise meaningful here.
        df = pl.DataFrame({"book_id": sorted(updated_ids)})
        local_path = self._save_locally(df, SCRAPED_IDS_FILENAME)
        self._upload(
            local_path, SCRAPED_IDS_FILENAME, f"Update book_ids tracker ({len(updated_ids)} total)"
        )
        return updated_ids

    def load(
        self, records: list[dict[str, Any]], filename: str, known_book_ids: set[str]
    ) -> tuple[pl.DataFrame, set[str]]:
        """Batch records into a DataFrame, save and push it, then update the scraped-ids tracker.

        This is the single checkpoint operation the pipeline calls once
        per batch: write the data file, upload it, then immediately
        update the tracker so that a crash right after this call still
        leaves the Hub in a consistent, resumable state.

        Args:
            records: List of final book record dicts to persist.
            filename: File name only for the data batch (e.g. "books-part7.parquet").
            known_book_ids: Full set of book_ids already known before this batch.

        Returns:
            Tuple of (the DataFrame written and uploaded, the updated set
            of all known book_ids including this batch).
        """
        df = pl.DataFrame(records)
        local_path = self._save_locally(df, filename)

        self.logger.info(f"Sinking {HF_RAW_DIR}/{filename}: uploading {df.height} row(s)...")
        self._upload(local_path, filename, f"Add {filename} ({df.height} books)")
        self.logger.info(f"Sunk {HF_RAW_DIR}/{filename}: {df.height} row(s) pushed to {HF_REPO_ID}")

        # Tracker update happens AFTER the data upload succeeds, so a
        # failure here still leaves the data file safely on the Hub even
        # if the tracker itself doesn't get updated this call.
        updated_ids = self._update_scraped_ids(df.get_column("book_id").to_list(), known_book_ids)
        self.logger.info(f"Tracker updated: {len(updated_ids)} book_id(s) now recorded")

        return df, updated_ids
