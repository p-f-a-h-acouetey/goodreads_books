"""Entry point: parses CLI args and triggers the pipeline via BookRunner.

Lives at the src/goodreads_etl package root -- outside extractors/,
loaders/, and pipelines/ -- since it is not itself an ETL stage, just the
thing you invoke to run one.

Usage:
    python -m src.goodreads_etl.main --sample-size 500000
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger

from src.goodreads_etl.pipelines.run import DEFAULT_SAMPLE_SIZE, BookRunner


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline run.

    Returns:
        Parsed arguments with sample_size.
    """
    parser = argparse.ArgumentParser(description="Run the Goodreads books ETL pipeline.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Target total number of valid book records the dataset should hold (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    return parser.parse_args()


async def _main() -> None:
    """Parse args, run the pipeline once, log the final shape."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    args = _parse_args()
    runner = BookRunner()
    await runner.run(args.sample_size)


if __name__ == "__main__":
    asyncio.run(_main())
