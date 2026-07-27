"""CLI entrypoint for the Goodreads book ETL pipeline."""

from __future__ import annotations

import argparse
import os
import sys

from loguru import logger

from src.goodreads_etl.pipelines.run_pipeline import scrape_books


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for the scrape run.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - `hf_token` (str | None): Token for Hugging Face authentication.
            - `log_level` (str): Logging severity level (e.g., "INFO", "DEBUG").
            - `resume` (bool): Flag indicating whether to resume from local state.
    """
    parser = argparse.ArgumentParser(
        description="Goodreads Book ETL Pipeline (random sampling, HF Hub output)"
    )
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the local scraped-ID tracker instead of starting fresh.",
    )
    return parser.parse_args()


def _configure_logging(*, log_level: str) -> None:
    """Reset and configure loguru sinks for CLI output.

    Removes existing sinks and adds a stderr sink with color support,
    asynchronous logging, and the specified severity filter level.

    Args:
        log_level: The minimum log level threshold to output (e.g., "INFO").
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper(), colorize=True, enqueue=True)


def main() -> None:
    """Parse CLI args, configure logging, and run the pipeline.

    Serves as the main script entrypoint when executing the pipeline from the shell.
    """
    args = _parse_args()
    _configure_logging(log_level=args.log_level)
    scrape_books(hf_token=args.hf_token, resume=args.resume)


if __name__ == "__main__":
    main()