"""Orchestrates Extract -> Transform -> Load in checkpoint-sized batches.

This is the only module that imports across all three ETL stage modules --
extract, transform, load -- keeping the dependency graph a strict DAG with
one composition point instead of scattered cross-imports.
"""

from __future__ import annotations

import asyncio

from huggingface_hub import HfApi
from loguru import logger

from src.goodreads_etl.extractors.extract import run_sampling_crawl
from src.goodreads_etl.loaders.load import (
    append_local_scraped_ids,
    ensure_repo_exists,
    get_hf_api,
    get_next_part_number,
    load_local_scraped_ids,
    push_checkpoint_to_hub,
    save_book_ids_to_hub,
)
from src.goodreads_etl.transformers.transform import records_to_dataframe
from src.goodreads_etl.utils.book_recorder import BookRecord
from src.goodreads_etl.utils.config_setter import SETTINGS, Settings


def _load_resume_state(*, resume: bool, settings: Settings) -> tuple[set[str], list[str]]:
    """Load already-scraped IDs if resuming, else start from empty state.

    Args:
        resume: Flag indicating whether to reload state from local storage.
        settings: Pipeline configuration settings containing paths.

    Returns:
        A tuple containing:
            - A set of already-tried book ID strings for fast lookup during crawl.
            - A list of all previously sampled book ID strings to maintain count state.
    """
    scraped_ids = load_local_scraped_ids(settings=settings) if resume else set()
    return set(scraped_ids), list(scraped_ids)


async def _run_one_batch(
    *,
    batch_size: int,
    already_tried_book_ids: set[str],
    settings: Settings,
) -> list[BookRecord]:
    """Run one sampling crawl batch and return its BookRecords.

    Args:
        batch_size: Number of unique book records to sample in this batch.
        already_tried_book_ids: Set of book IDs to skip during sampling.
        settings: Pipeline configuration settings.

    Returns:
        A list of extracted `BookRecord` instances from the crawl batch.
    """
    return await run_sampling_crawl(
        sample_size=batch_size,
        min_book_id=settings.min_book_id,
        max_book_id=settings.max_book_id,
        settings=settings,
        already_tried_book_ids=already_tried_book_ids,
    )


def _persist_batch(
    *,
    records: list[BookRecord],
    part_number: int,
    api: HfApi,
    settings: Settings,
) -> list[str]:
    """Transform one batch to a DataFrame, push it, and update local tracker.

    Args:
        records: A list of extracted `BookRecord` instances to persist.
        part_number: The sequential checkpoint part index.
        api: An authenticated `HfApi` client instance.
        settings: Pipeline configuration settings.

    Returns:
        A list of string book IDs included in this persisted batch.
    """
    dataframe = records_to_dataframe(records=records)
    batch_ids = [record.book_id for record in records]

    push_checkpoint_to_hub(dataframe=dataframe, part_number=part_number, api=api, settings=settings)
    append_local_scraped_ids(book_ids=batch_ids, settings=settings)

    return batch_ids


def scrape_books(
    *,
    hf_token: str | None = None,
    resume: bool = False,
    settings: Settings = SETTINGS,
) -> None:
    """Sample Goodreads books in batches and push checkpoints to the Hub.

    Orchestrates the full ETL flow: sets up the repository, manages batch execution,
    transforms output to Polars DataFrames, pushes parquet checkpoints, updates
    local state tracking, and uploads the aggregated ID manifest upon completion.

    This is the single public entrypoint for the whole pipeline.

    Args:
        hf_token: Optional Hugging Face access token for authentication. Defaults
            to None (resolved from `HF_TOKEN` environment variable).
        resume: Whether to resume from local tracking files (`True`) or start a fresh
            run from scratch (`False`). Defaults to `False`.
        settings: Pipeline settings and parameters. Defaults to `SETTINGS`.
    """
    api = get_hf_api(hf_token=hf_token)
    ensure_repo_exists(api=api, repo_id=settings.repo_id)

    already_tried_book_ids, all_sampled_ids = _load_resume_state(resume=resume, settings=settings)
    remaining = max(settings.sample_size - len(all_sampled_ids), 0)

    logger.info(
        "Starting sampling: target={}, already have={}, remaining={}",
        settings.sample_size,
        len(all_sampled_ids),
        remaining,
    )

    if remaining == 0:
        logger.info("Nothing to scrape: resume data already satisfies target.")
        save_book_ids_to_hub(book_ids=all_sampled_ids, api=api, settings=settings)
        return

    part_number = get_next_part_number(api=api, settings=settings)

    while remaining > 0:
        batch_size = min(settings.checkpoint_every, remaining)

        records = asyncio.run(
            _run_one_batch(
                batch_size=batch_size,
                already_tried_book_ids=already_tried_book_ids,
                settings=settings,
            )
        )

        if not records:
            logger.error("Crawl batch returned zero records; stopping early.")
            break

        batch_ids = _persist_batch(records=records, part_number=part_number, api=api, settings=settings)

        already_tried_book_ids.update(batch_ids)
        all_sampled_ids.extend(batch_ids)
        remaining -= len(records)
        part_number += 1

        logger.info(
            "Batch complete: scraped={}, remaining={}, total_collected={}",
            len(records),
            remaining,
            len(all_sampled_ids),
        )

    save_book_ids_to_hub(book_ids=all_sampled_ids, api=api, settings=settings)
    logger.info("Sampling complete: collected={} target={}", len(all_sampled_ids), settings.sample_size)