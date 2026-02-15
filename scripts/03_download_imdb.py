#!/usr/bin/env python3
"""Download and prepare IMDB public dataset files.

Downloads TSV files from IMDB's public dataset, extracts them, and
optionally filters to relevant columns to reduce file sizes.
"""

import argparse
import gzip
import logging
import shutil
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

# Columns to keep when filtering (reduces memory/disk usage significantly)
COLUMN_FILTERS: dict[str, list[str]] = {
    "name.basics.tsv": [
        "nconst",
        "primaryName",
        "birthYear",
        "primaryProfession",
        "knownForTitles",
    ],
    "title.basics.tsv": [
        "tconst",
        "titleType",
        "primaryTitle",
        "originalTitle",
        "startYear",
        "genres",
    ],
    "title.crew.tsv": [
        "tconst",
        "directors",
        "writers",
    ],
    "title.principals.tsv": [
        "tconst",
        "nconst",
        "category",
        "characters",
    ],
}


def download_file(url: str, output_path: Path, overwrite: bool = False) -> Path:
    """Download a file with a progress bar.

    Args:
        url: URL to download from.
        output_path: Local path to save the file.
        overwrite: If True, re-download even if the file exists.

    Returns:
        Path to the downloaded file.
    """
    if output_path.exists() and not overwrite:
        logger.info("Already downloaded: %s", output_path.name)
        return output_path

    logger.info("Downloading: %s", url)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with (
        open(output_path, "wb") as f,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=output_path.name,
            leave=True,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info("Saved: %s", output_path.name)
    return output_path


def extract_gz(gz_path: Path, output_path: Path, overwrite: bool = False) -> Path:
    """Extract a .gz file.

    Args:
        gz_path: Path to the gzipped file.
        output_path: Path for the extracted output.
        overwrite: If True, re-extract even if the output exists.

    Returns:
        Path to the extracted file.
    """
    if output_path.exists() and not overwrite:
        logger.info("Already extracted: %s", output_path.name)
        return output_path

    logger.info("Extracting: %s -> %s", gz_path.name, output_path.name)
    with gzip.open(gz_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    logger.info("Extracted: %s", output_path.name)
    return output_path


def filter_tsv(input_path: Path, columns: list[str]) -> Path:
    """Filter a TSV file to keep only specified columns, overwriting in place.

    Args:
        input_path: Path to the TSV file.
        columns: List of column names to keep.

    Returns:
        Path to the filtered file.
    """
    filtered_path = input_path.with_suffix(".filtered.tsv")
    logger.info("Filtering %s to columns: %s", input_path.name, columns)

    with (
        open(input_path, "r", encoding="utf-8") as f_in,
        open(filtered_path, "w", encoding="utf-8") as f_out,
    ):
        header = f_in.readline().strip().split("\t")
        # Find indices of columns to keep
        indices = []
        for col in columns:
            if col in header:
                indices.append(header.index(col))
            else:
                logger.warning("Column '%s' not found in %s", col, input_path.name)

        if not indices:
            logger.error("No valid columns found for %s", input_path.name)
            return input_path

        # Write filtered header
        kept_cols = [header[i] for i in indices]
        f_out.write("\t".join(kept_cols) + "\n")

        # Process rows
        row_count = 0
        for line in f_in:
            parts = line.strip().split("\t")
            filtered = []
            for i in indices:
                filtered.append(parts[i] if i < len(parts) else "")
            f_out.write("\t".join(filtered) + "\n")
            row_count += 1

    # Replace original with filtered version
    filtered_path.replace(input_path)
    logger.info("Filtered %s: kept %d columns, %d rows", input_path.name, len(indices), row_count)
    return input_path


def main() -> None:
    """Download and extract IMDB dataset files."""
    parser = argparse.ArgumentParser(
        description="Download IMDB public dataset files."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and re-extract existing files",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Keep all columns (don't filter to relevant subset)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        choices=[f.replace(".gz", "") for f in config.IMDB_FILES],
        help="Only download specific files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    config.IMDB_DIR.mkdir(parents=True, exist_ok=True)

    files_to_download = config.IMDB_FILES
    if args.files:
        files_to_download = [f + ".gz" for f in args.files]

    for filename in files_to_download:
        url = config.IMDB_BASE_URL + filename
        gz_path = config.IMDB_DIR / filename
        tsv_path = config.IMDB_DIR / filename.replace(".gz", "")

        # Download
        try:
            download_file(url, gz_path, overwrite=args.overwrite)
        except requests.RequestException as e:
            logger.error("Failed to download %s: %s", filename, e)
            continue

        # Extract
        extract_gz(gz_path, tsv_path, overwrite=args.overwrite)

        # Filter columns
        tsv_name = filename.replace(".gz", "")
        if not args.no_filter and tsv_name in COLUMN_FILTERS:
            filter_tsv(tsv_path, COLUMN_FILTERS[tsv_name])

        # Remove .gz to save space
        if gz_path.exists():
            gz_path.unlink()
            logger.info("Removed compressed file: %s", gz_path.name)

    logger.info("IMDB dataset download complete.")


if __name__ == "__main__":
    main()
