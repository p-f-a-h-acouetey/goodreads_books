"""LOAD stage: Hugging Face Hub I/O and scraped-book-ID bookkeeping.

Exposes the public functions the pipeline needs: repo setup, checkpoint
push, and ID tracking (local file + Hub-hosted file), consolidated into
one module since they're all "where does the data end up" concerns.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from loguru import logger

from src.goodreads_etl.utils.config_setter import SETTINGS, Settings

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def _resolve_hf_token(*, explicit_token: str | None) -> str | None:
    """Resolve a Hugging Face token.

    Checks the explicitly passed token first, falling back to the `HF_TOKEN`
    environment variable if non-existent or None.

    Args:
        explicit_token: An optional token string passed by the caller.

    Returns:
        The resolved token string, or None if no token is available.
    """
    return explicit_token or os.getenv("HF_TOKEN")


def get_hf_api(*, hf_token: str | None = None) -> HfApi:
    """Build an authenticated HfApi client.

    Public: used by the pipeline to obtain one client shared across the whole run.

    Args:
        hf_token: Optional Hugging Face API token. Defaults to None (resolved
            from environment).

    Returns:
        An authenticated `HfApi` instance.
    """
    return HfApi(token=_resolve_hf_token(explicit_token=hf_token))


# ---------------------------------------------------------------------------
# Repo setup
# ---------------------------------------------------------------------------


def ensure_repo_exists(*, api: HfApi, repo_id: str = SETTINGS.repo_id) -> None:
    """Create the target dataset repo if it does not already exist.

    Args:
        api: An authenticated `HfApi` client instance.
        repo_id: The target Hugging Face repository ID (e.g., "org/dataset-name").
            Defaults to `SETTINGS.repo_id`.
    """
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)


# ---------------------------------------------------------------------------
# Checkpoint numbering and upload
# ---------------------------------------------------------------------------


def _extract_part_numbers(*, files: list[str], filename_template: str) -> list[int]:
    """Parse existing checkpoint filenames to find their part numbers.

    Args:
        files: List of file names retrieved from the repository.
        filename_template: Template pattern string containing `{part}`.

    Returns:
        A list of integer part numbers extracted from matching filenames.
    """
    pattern = re.escape(filename_template).replace(r"\{part\}", r"(\d+)")
    return [
        int(match.group(1)) for file_name in files if (match := re.match(f"^{pattern}$", file_name))
    ]


def get_next_part_number(*, api: HfApi, settings: Settings = SETTINGS) -> int:
    """Determine the next checkpoint part number from existing repo files.

    Args:
        api: An authenticated `HfApi` client instance.
        settings: Pipeline configuration settings. Defaults to `SETTINGS`.

    Returns:
        The next integer part number (starts at 1 if no files exist).
    """
    try:
        files = api.list_repo_files(repo_id=settings.repo_id, repo_type="dataset")
    except Exception as exc:
        logger.warning("Could not list repo files, defaulting to part 1: {}", exc)
        return 1

    part_numbers = _extract_part_numbers(
        files=files, filename_template=settings.checkpoint_filename_template
    )
    return max(part_numbers, default=0) + 1


def push_checkpoint_to_hub(
    *,
    dataframe: pl.DataFrame,
    part_number: int,
    api: HfApi,
    settings: Settings = SETTINGS,
) -> None:
    """Serialize a DataFrame to Parquet and upload it as one checkpoint file.

    Args:
        dataframe: The Polars DataFrame to push. No action taken if empty.
        part_number: The incremental part number to substitute into the template.
        api: An authenticated `HfApi` client instance.
        settings: Pipeline configuration settings. Defaults to `SETTINGS`.
    """
    if dataframe.is_empty():
        return

    filename = settings.checkpoint_filename_template.format(part=part_number)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / filename
        dataframe.write_parquet(tmp_path)

        api.upload_file(
            path_or_fileobj=str(tmp_path),
            path_in_repo=filename,
            repo_id=settings.repo_id,
            repo_type="dataset",
            commit_message=f"Add {filename} ({dataframe.height} books)",
        )

    logger.info("Pushed {} books to {}/{}", dataframe.height, settings.repo_id, filename)


# ---------------------------------------------------------------------------
# Local scraped-ID tracker
# ---------------------------------------------------------------------------


def load_local_scraped_ids(*, settings: Settings = SETTINGS) -> set[str]:
    """Load already-scraped book IDs from the local tracker file.

    Args:
        settings: Pipeline configuration settings. Defaults to `SETTINGS`.

    Returns:
        A set of string book IDs previously recorded in the local tracking file.
    """
    file_path = Path(settings.local_scraped_ids_path)
    if not file_path.exists():
        return set()

    with file_path.open("r", encoding=settings.encoding) as file:
        return {line.strip() for line in file if line.strip()}


def append_local_scraped_ids(*, book_ids: list[str], settings: Settings = SETTINGS) -> None:
    """Append newly scraped book IDs to the local tracker file.

    Called right after each successful checkpoint push, so a crash
    mid-run never loses progress that already made it to the Hub.

    Args:
        book_ids: List of newly processed book ID strings to record.
        settings: Pipeline configuration settings. Defaults to `SETTINGS`.
    """
    if not book_ids:
        return

    file_path = Path(settings.local_scraped_ids_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("a", encoding=settings.encoding) as file:
        file.writelines(f"{book_id}\n" for book_id in book_ids)

    logger.info(
        "Appended {} book IDs to local tracker {}", len(book_ids), settings.local_scraped_ids_path
    )


# ---------------------------------------------------------------------------
# Hub-hosted scraped-ID tracker
# ---------------------------------------------------------------------------


def _load_hub_scraped_ids(*, api: HfApi, settings: Settings = SETTINGS) -> set[str]:
    """Download the Hub-hosted scraped-ID file, if it exists.

    Args:
        api: An authenticated `HfApi` client instance.
        settings: Pipeline configuration settings. Defaults to `SETTINGS`.

    Returns:
        A set of string book IDs stored in the remote Hugging Face repository tracker.
    """
    try:
        path = hf_hub_download(
            repo_id=settings.repo_id,
            filename=settings.book_ids_filename,
            repo_type="dataset",
            token=api.token,
        )
    except EntryNotFoundError:
        return set()
    except Exception as exc:
        logger.warning("Could not download Hub ID tracker: {}", exc)
        return set()

    with open(path, encoding=settings.encoding) as file:
        return {line.strip() for line in file if line.strip()}


def save_book_ids_to_hub(*, book_ids: list[str], api: HfApi, settings: Settings = SETTINGS) -> None:
    """Merge new IDs with existing Hub IDs, then upload the combined list.

    Args:
        book_ids: List of new book ID strings to merge with existing remote IDs.
        api: An authenticated `HfApi` client instance.
        settings: Pipeline configuration settings. Defaults to `SETTINGS`.
    """
    if not book_ids:
        return

    existing_ids = _load_hub_scraped_ids(api=api, settings=settings)
    combined_ids = list(dict.fromkeys([*existing_ids, *book_ids]))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / settings.book_ids_filename
        tmp_path.write_text("\n".join(combined_ids), encoding=settings.encoding)

        api.upload_file(
            path_or_fileobj=str(tmp_path),
            path_in_repo=settings.book_ids_filename,
            repo_id=settings.repo_id,
            repo_type="dataset",
            commit_message=f"Update {settings.book_ids_filename} with {len(book_ids)} new IDs",
        )

    logger.info(
        "Saved {} total book IDs to {}/{}",
        len(combined_ids),
        settings.repo_id,
        settings.book_ids_filename,
    )
