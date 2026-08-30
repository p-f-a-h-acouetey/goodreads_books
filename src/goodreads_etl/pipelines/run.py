"""PIPELINE stage: BookRunner orchestrates EXTRACT and LOAD to hit a sample-size target.

Ties BookExtractor (src/goodreads_etl/extractors/extract.py) and
BookLoader (src/goodreads_etl/loaders/load.py) together:

    1. Ask BookLoader how many books are already known (Hub tracker).
    2. If the target is already met, do nothing.
    3. Otherwise ask BookExtractor for however many more books are needed.
    4. Hand the results back to BookLoader to persist in checkpointed
       batches, pushing each batch to the Hugging Face Hub and updating
       the scraped-ID tracker right after each push (so a crash mid-run
       never loses progress that already made it to the Hub).

Exposes exactly one class, BookRunner, with one public method:
- run(sample_size)
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from src.goodreads_etl.extractors.extract import BookExtractor
from src.goodreads_etl.loaders.load import PARQUET_BATCH_SIZE, PARQUET_FILENAME_TEMPLATE, BookLoader

# --- Sampling defaults ---------------------------------------------------------
DEFAULT_SAMPLE_SIZE = 500_000


class BookRunner:
    """Orchestrates BookExtractor and BookLoader to hit a sample_size target."""

    def __init__(self) -> None:
        self.logger = logger.bind(component="BookRunner")
        self.extractor = BookExtractor()
        self.loader = BookLoader()

    async def run(self, sample_size: int = DEFAULT_SAMPLE_SIZE) -> pl.DataFrame | None:
        """Run the ETL pipeline until the dataset holds sample_size valid book records total.

        sample_size is a TARGET, not an increment -- if the tracker already
        holds sample_size or more book_ids, this returns immediately.
        Otherwise it runs one queue-based sampling crawl for the remaining
        count, then flushes results to the Hub in PARQUET_BATCH_SIZE chunks.

        Args:
            sample_size: Target total number of valid book records the
                dataset should hold.

        Returns:
            A DataFrame of the records collected in THIS run; empty if the
            target was already met before this call.
        """
        self.logger.info(f"Starting pipeline -- target {sample_size} total valid books...")

        known_book_ids = self.loader.load_scraped_ids()

        if len(known_book_ids) >= sample_size:
            self.logger.info(
                f"Target already met ({len(known_book_ids)}/{sample_size} books already scraped), "
                "nothing to do"
            )
            return None

        remaining_target = sample_size - len(known_book_ids)
        self.logger.info(f"{len(known_book_ids)} books already scraped, collecting {remaining_target} more")

        part_number = self.loader.get_next_part_number()
        tried_book_ids: set[str] = set(known_book_ids)

        new_records = await self.extractor.extract(remaining_target, tried_book_ids)

        all_batches: list[pl.DataFrame] = []
        for start in range(0, len(new_records), PARQUET_BATCH_SIZE):
            batch = new_records[start : start + PARQUET_BATCH_SIZE]
            batch_df, known_book_ids = self.loader.load(
                batch, PARQUET_FILENAME_TEMPLATE.format(part=part_number), known_book_ids
            )
            all_batches.append(batch_df)
            part_number += 1

        self.logger.info(f"Pipeline complete -- {len(known_book_ids)}/{sample_size} total books now in the dataset")
        return None
