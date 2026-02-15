#!/usr/bin/env python3
"""Download podcast episodes from The Rewatchables RSS feed.

Parses the RSS feed, extracts episode metadata, and downloads audio files
to the local data directory.
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import feedparser
import requests
from tqdm import tqdm

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")[:120]


def parse_feed(feed_url: str) -> list[dict]:
    """Parse an RSS feed and return a list of episode metadata dicts.

    Args:
        feed_url: URL of the RSS feed to parse.

    Returns:
        List of dicts with keys: title, date, description, audio_url, slug.
    """
    logger.info("Fetching RSS feed: %s", feed_url)
    feed = feedparser.parse(feed_url)

    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Failed to parse RSS feed: {feed.bozo_exception}")

    episodes = []
    for entry in feed.entries:
        # Find the audio enclosure
        audio_url = None
        for link in entry.get("links", []):
            if link.get("type", "").startswith("audio/"):
                audio_url = link["href"]
                break
        # Fallback: check enclosures
        if not audio_url:
            for enc in entry.get("enclosures", []):
                if enc.get("type", "").startswith("audio/"):
                    audio_url = enc["href"]
                    break

        if not audio_url:
            logger.warning("No audio URL found for entry: %s", entry.get("title"))
            continue

        # Parse publication date
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        if published:
            date_str = datetime(*published[:6]).strftime("%Y-%m-%d")
        else:
            date_str = "unknown-date"

        title = entry.get("title", "Untitled")
        slug = f"{date_str}_{slugify(title)}"

        episodes.append({
            "title": title,
            "date": date_str,
            "description": entry.get("summary", ""),
            "audio_url": audio_url,
            "slug": slug,
        })

    logger.info("Found %d episodes in feed", len(episodes))
    return episodes


def download_episode(episode: dict, output_dir: Path, overwrite: bool = False) -> Path:
    """Download a single episode's audio file.

    Args:
        episode: Episode metadata dict with audio_url and slug keys.
        output_dir: Directory to save the audio file.
        overwrite: If True, re-download even if the file exists.

    Returns:
        Path to the downloaded audio file.
    """
    # Determine file extension from URL (default to .mp3)
    url = episode["audio_url"]
    ext = ".mp3"
    url_path = url.split("?")[0]
    if "." in url_path.split("/")[-1]:
        ext = "." + url_path.split("/")[-1].rsplit(".", 1)[-1]

    output_path = output_dir / f"{episode['slug']}{ext}"

    if output_path.exists() and not overwrite:
        logger.info("Already downloaded: %s", output_path.name)
        return output_path

    logger.info("Downloading: %s", episode["title"])
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with (
        open(output_path, "wb") as f,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=episode["slug"][:50],
            leave=True,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info("Saved: %s", output_path.name)
    return output_path


def save_metadata(episodes: list[dict], output_path: Path) -> None:
    """Save episode metadata list to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
    logger.info("Saved metadata for %d episodes to %s", len(episodes), output_path)


def main() -> None:
    """Parse CLI args, fetch RSS feed, and download episodes."""
    parser = argparse.ArgumentParser(
        description="Download episodes from The Rewatchables RSS feed."
    )
    parser.add_argument(
        "-n", "--num-episodes",
        type=int,
        default=config.DEFAULT_EPISODE_LIMIT,
        help=f"Number of most recent episodes to download (default: {config.DEFAULT_EPISODE_LIMIT})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available episodes",
    )
    parser.add_argument(
        "--feed-url",
        default=config.RSS_FEED_URL,
        help="RSS feed URL (default: The Rewatchables)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only save metadata, skip audio downloads",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    # Ensure output directories exist
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Parse feed
    all_episodes = parse_feed(args.feed_url)

    # Limit episodes
    if args.all:
        episodes = all_episodes
    else:
        episodes = all_episodes[: args.num_episodes]

    logger.info("Processing %d episode(s)", len(episodes))

    # Save metadata
    save_metadata(episodes, config.EPISODES_METADATA_PATH)

    if args.metadata_only:
        logger.info("Metadata-only mode; skipping audio downloads.")
        return

    # Download audio
    for episode in episodes:
        try:
            download_episode(episode, config.AUDIO_DIR, overwrite=args.overwrite)
        except requests.RequestException as e:
            logger.error("Failed to download '%s': %s", episode["title"], e)


if __name__ == "__main__":
    main()
