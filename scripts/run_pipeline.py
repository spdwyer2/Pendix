#!/usr/bin/env python3
"""Run the Pendix data pipeline.

Orchestrates all pipeline steps in sequence, with support for running
individual steps or resuming from a specific step.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent

STEPS: dict[str, dict] = {
    "download": {
        "script": "01_download_episodes.py",
        "description": "Download podcast episodes from RSS feed",
    },
    "transcribe": {
        "script": "02_transcribe.py",
        "description": "Transcribe audio files using Whisper",
    },
    "imdb": {
        "script": "03_download_imdb.py",
        "description": "Download IMDB dataset files",
    },
    "entities": {
        "script": "04_extract_entities.py",
        "description": "Extract entities from transcripts",
    },
    "index": {
        "script": "05_index_elasticsearch.py",
        "description": "Index data into Elasticsearch",
    },
}

STEP_ORDER = ["download", "transcribe", "imdb", "entities", "index"]


def run_step(step_name: str, extra_args: list[str] | None = None) -> bool:
    """Run a single pipeline step.

    Args:
        step_name: Name of the step to run.
        extra_args: Additional CLI arguments to pass to the script.

    Returns:
        True if the step completed successfully, False otherwise.
    """
    step = STEPS[step_name]
    script_path = SCRIPTS_DIR / step["script"]

    logger.info("=" * 60)
    logger.info("STEP: %s — %s", step_name.upper(), step["description"])
    logger.info("Script: %s", script_path.name)
    logger.info("=" * 60)

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error("Step '%s' failed with exit code %d", step_name, result.returncode)
        return False

    logger.info("Step '%s' completed successfully.", step_name)
    return True


def main() -> None:
    """Parse CLI args and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the Pendix data pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps (in order):
  download    — Download podcast episodes from RSS feed
  transcribe  — Transcribe audio files using Whisper
  imdb        — Download IMDB dataset files
  entities    — Extract entities from transcripts
  index       — Index data into Elasticsearch

Examples:
  python run_pipeline.py                    # Run all steps
  python run_pipeline.py --step download    # Run only the download step
  python run_pipeline.py --from transcribe  # Run from transcribe onward
  python run_pipeline.py --step download -- --num-episodes 5
        """,
    )
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        help="Run only a specific step",
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        choices=list(STEPS.keys()),
        help="Start from a specific step and run all subsequent steps",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        default=True,
        help="Stop the pipeline if a step fails (default: True)",
    )
    parser.add_argument(
        "--no-stop-on-error",
        action="store_false",
        dest="stop_on_error",
        help="Continue running subsequent steps even if one fails",
    )

    # Allow passing extra args to individual scripts via '--'
    args, extra_args = parser.parse_known_args()
    # Remove leading '--' if present
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    # Determine which steps to run
    if args.step:
        steps_to_run = [args.step]
    elif args.from_step:
        start_idx = STEP_ORDER.index(args.from_step)
        steps_to_run = STEP_ORDER[start_idx:]
    else:
        steps_to_run = STEP_ORDER

    logger.info("Pendix Data Pipeline")
    logger.info("Steps to run: %s", ", ".join(steps_to_run))
    logger.info("")

    # Run steps
    results: dict[str, bool] = {}
    for step_name in steps_to_run:
        # Only pass extra_args if running a single step
        step_extra = extra_args if args.step else None
        success = run_step(step_name, step_extra)
        results[step_name] = success

        if not success and args.stop_on_error:
            logger.error("Pipeline stopped due to failure in step '%s'.", step_name)
            break

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for step_name, success in results.items():
        status = "OK" if success else "FAILED"
        logger.info("  %-12s %s", step_name, status)

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        logger.error("Pipeline completed with %d failure(s).", len(failed))
        sys.exit(1)
    else:
        logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
