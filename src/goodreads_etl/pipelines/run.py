"""PIPELINE stage: BookRunner orchestrates EXTRACT and LOAD to hit a sample-size target.

Ties BookExtractor (src/goodreads_etl/extractors/extract.py) and
BookLoader (src/goodreads_etl/loaders/load.py) together:

    1. Ask BookLoader how many books are already known (Hub tracker).
    2. If the target is already met, do nothing.
    3. Otherwise, repeatedly ask BookExtractor for PARQUET_BATCH_SIZE
       books at a time (not the full remaining target in one call), so
       each chunk can be checkpointed to the Hub as soon as it's ready.
    4. Hand each chunk back to BookLoader to persist immediately,
       pushing it to the Hugging Face Hub and updating the scraped-ID
       tracker right after (so a crash mid-run never loses progress
       that already made it to the Hub).

Calling extractor.extract() once with the full remaining target would
block until that entire count is collected before any checkpoint could
be written -- for large targets (e.g. 500,000) this means nothing ever
gets saved, since the crawl never finishes in a single run. Chunking the
calls here, instead, is what makes checkpointing actually happen.

Exposes exactly one class, BookRunner, with one public method:
- run(sample_size)
"""

from __future__ import annotations

from loguru import logger

from src.goodreads_etl.extractors.extract import BookExtractor
from src.goodreads_etl.loaders.load import PARQUET_FILENAME_TEMPLATE, BookLoader

# --- Sampling defaults ---------------------------------------------------------
DEFAULT_SAMPLE_SIZE = 500_000

# --- Checkpoint batching ---------------------------------------------------------
# Lives here (not in load.py) because chunk sizing is a pipeline-orchestration
# decision -- how many records BookRunner asks BookExtractor for at a time,
# before handing that chunk to BookLoader to persist. BookLoader itself is
# agnostic to batch size; it just persists whatever it's given.
PARQUET_BATCH_SIZE = 5_000


class BookRunner:
    """Orchestrates BookExtractor and BookLoader to hit a sample_size target."""

    def __init__(self) -> None:
        self.logger = logger.bind(component="BookRunner")
        self.extractor = BookExtractor()
        self.loader = BookLoader()

    async def run(self, sample_size: int = DEFAULT_SAMPLE_SIZE) -> None:
        """Run the ETL pipeline until the dataset holds sample_size valid book records total.

        sample_size is a TARGET, not an increment -- if the tracker already
        holds sample_size or more book_ids, this returns immediately.
        Otherwise it runs repeated chunked crawls (PARQUET_BATCH_SIZE books
        each) until the remaining target is met, checkpointing each chunk
        to the Hub as soon as it's collected.

        Args:
            sample_size: Target total number of valid book records the
                dataset should hold.

        Returns:
            None
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
        self.logger.info(
            f"{len(known_book_ids)} books already scraped, collecting {remaining_target} more"
        )

        part_number = self.loader.get_next_part_number()
        # tried_book_ids is passed by reference into extractor.extract()
        # and mutated in place there with every ID drawn -- reusing the
        # SAME set object across every chunk below (not rebuilding it) is
        # what stops later chunks from re-drawing IDs already tried earlier
        # in this same run.
        tried_book_ids: set[str] = set(known_book_ids)

        collected_so_far = 0

        while collected_so_far < remaining_target:
            # Never ask for more than PARQUET_BATCH_SIZE in one extract()
            # call (so each chunk can be checkpointed promptly), and never
            # ask for more than what's still needed to hit remaining_target.
            chunk_target = min(PARQUET_BATCH_SIZE, remaining_target - collected_so_far)

            self.logger.info(
                f"Extracting chunk of up to {chunk_target} book(s) "
                f"({collected_so_far}/{remaining_target} collected so far)..."
            )
            chunk_records = await self.extractor.extract(chunk_target, tried_book_ids)

            if not chunk_records:
                # extract() returns an empty list when the untried ID space
                # is exhausted -- no point looping further this run.
                self.logger.warning(
                    "Extractor returned no records this chunk -- stopping "
                    "(ID space likely exhausted)"
                )
                break

            # Checkpoint immediately: persist locally, push to the Hub, and
            # update the tracker, all before starting the next chunk -- this
            # is what makes progress durable across a crash mid-run.
            _, known_book_ids = self.loader.load(
                chunk_records, PARQUET_FILENAME_TEMPLATE.format(part=part_number), known_book_ids
            )
            part_number += 1
            collected_so_far += len(chunk_records)

        self.logger.info(
            f"Pipeline complete -- {len(known_book_ids)}/{sample_size} total books now in the dataset"  # NOQA E501
        )

        # NOTE: run() intentionally returns None -- callers (e.g. main.py)
        # don't need the collected data back in memory, since every batch
        # was already checkpointed to the Hub as it was produced above.
        return None
