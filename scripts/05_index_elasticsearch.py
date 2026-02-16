#!/usr/bin/env python3
"""Index transcripts, IMDB data, and entity mentions into Elasticsearch.

Creates indices using predefined mappings and bulk-indexes episode transcripts,
IMDB people, and cross-referenced entity mentions.
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

MAPPINGS_DIR = config.PROJECT_ROOT / "elasticsearch" / "mappings"


def get_es_client(url: str, api_key: str = "") -> Elasticsearch:
    """Create and verify an Elasticsearch client connection.

    Args:
        url: Elasticsearch URL.
        api_key: Optional base64-encoded API key for Elastic Cloud.

    Returns:
        Connected Elasticsearch client.
    """
    kwargs: dict = {"hosts": [url]}
    if api_key:
        kwargs["api_key"] = api_key
    es = Elasticsearch(**kwargs)
    if not es.ping():
        raise ConnectionError(f"Cannot connect to Elasticsearch at {url}")
    logger.info("Connected to Elasticsearch at %s", url)
    return es


def create_index(es: Elasticsearch, index_name: str, mapping_file: Path) -> None:
    """Create an ES index from a mapping file, deleting any existing one.

    Args:
        es: Elasticsearch client.
        index_name: Name for the index.
        mapping_file: Path to the JSON mapping file.
    """
    if es.indices.exists(index=index_name):
        logger.info("Deleting existing index: %s", index_name)
        es.indices.delete(index=index_name)

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    es.indices.create(index=index_name, body=mapping)
    logger.info("Created index: %s", index_name)


def index_episodes(es: Elasticsearch) -> int:
    """Index episode transcripts into Elasticsearch.

    Args:
        es: Elasticsearch client.

    Returns:
        Number of episodes indexed.
    """
    transcript_files = sorted(config.TRANSCRIPTS_DIR.glob("*.json"))
    if not transcript_files:
        logger.warning("No transcript files found in %s", config.TRANSCRIPTS_DIR)
        return 0

    # Load episode metadata for descriptions
    episode_meta = {}
    if config.EPISODES_METADATA_PATH.exists():
        with open(config.EPISODES_METADATA_PATH, "r", encoding="utf-8") as f:
            for ep in json.load(f):
                episode_meta[ep["slug"]] = ep

    actions = []
    for tf in tqdm(transcript_files, desc="Preparing episodes", unit="file"):
        with open(tf, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        slug = tf.stem
        meta = episode_meta.get(slug, {})

        doc = {
            "episode_title": transcript.get("episode_title", ""),
            "episode_date": transcript.get("episode_date", "") or None,
            "audio_file": transcript.get("audio_file", ""),
            "slug": slug,
            "description": meta.get("description", ""),
            "full_text": transcript.get("full_text", ""),
            "segments": transcript.get("segments", []),
        }

        # Skip docs with invalid dates
        if doc["episode_date"] and doc["episode_date"] == "unknown-date":
            doc["episode_date"] = None

        actions.append({
            "_index": config.ES_INDEX_EPISODES,
            "_id": slug,
            "_source": doc,
        })

    if actions:
        success, errors = bulk(es, actions, raise_on_error=False)
        logger.info("Indexed %d episodes (%d errors)", success, len(errors) if isinstance(errors, list) else 0)
        return success
    return 0


def index_people(es: Elasticsearch) -> int:
    """Index IMDB people data enriched with episode mention counts.

    Args:
        es: Elasticsearch client.

    Returns:
        Number of people indexed.
    """
    # First, load entity mentions to know which people are relevant
    combined_path = config.ENTITIES_DIR / "all_mentions.json"
    if not combined_path.exists():
        logger.warning("No all_mentions.json found. Run 04_extract_entities.py first.")
        return 0

    with open(combined_path, "r", encoding="utf-8") as f:
        all_episodes = json.load(f)

    # Build a map: nconst -> list of episode mentions
    nconst_mentions: dict[str, list[dict]] = defaultdict(list)
    nconst_info: dict[str, dict] = {}

    for episode in all_episodes:
        for person in episode.get("people", []):
            nc = person["nconst"]
            nconst_mentions[nc].append({
                "episode_title": episode.get("episode_title", ""),
                "episode_date": episode.get("episode_date", "") or None,
                "mention_count": person.get("mention_count", 0),
            })
            if nc not in nconst_info:
                nconst_info[nc] = person

    if not nconst_info:
        logger.warning("No entity mentions found. Nothing to index for people.")
        return 0

    # Also load full IMDB data for richer records
    names_file = config.IMDB_DIR / "name.basics.tsv"
    imdb_lookup: dict[str, dict] = {}
    if names_file.exists():
        with open(names_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                nc = row.get("nconst", "")
                if nc in nconst_info:
                    imdb_lookup[nc] = row

    actions = []
    for nc, info in tqdm(nconst_info.items(), desc="Preparing people", unit="person"):
        mentions = nconst_mentions[nc]
        # Clean up episode_date values
        for m in mentions:
            if m["episode_date"] == "unknown-date":
                m["episode_date"] = None

        imdb_row = imdb_lookup.get(nc, {})
        professions = info.get("professions", [])

        doc = {
            "nconst": nc,
            "name": info.get("name", imdb_row.get("primaryName", "")),
            "birth_year": imdb_row.get("birthYear", info.get("birth_year", "")),
            "professions": professions,
            "category": info.get("category", "other"),
            "known_for_titles": info.get("known_for_titles", []),
            "episode_mentions": mentions,
            "total_mentions": sum(m["mention_count"] for m in mentions),
        }

        actions.append({
            "_index": config.ES_INDEX_PEOPLE,
            "_id": nc,
            "_source": doc,
        })

    if actions:
        success, errors = bulk(es, actions, raise_on_error=False)
        logger.info("Indexed %d people (%d errors)", success, len(errors) if isinstance(errors, list) else 0)
        return success
    return 0


def main() -> None:
    """Create indices and index all data into Elasticsearch."""
    parser = argparse.ArgumentParser(
        description="Index data into Elasticsearch."
    )
    parser.add_argument(
        "--es-url",
        default=config.ELASTICSEARCH_URL,
        help=f"Elasticsearch URL (default: {config.ELASTICSEARCH_URL})",
    )
    parser.add_argument(
        "--index",
        choices=["episodes", "people", "all"],
        default="all",
        help="Which index to create and populate (default: all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    es = get_es_client(args.es_url, api_key=config.ES_API_KEY)

    if args.index in ("episodes", "all"):
        create_index(es, config.ES_INDEX_EPISODES, MAPPINGS_DIR / "episodes.json")
        count = index_episodes(es)
        logger.info("Episodes indexed: %d", count)

    if args.index in ("people", "all"):
        create_index(es, config.ES_INDEX_PEOPLE, MAPPINGS_DIR / "imdb_people.json")
        count = index_people(es)
        logger.info("People indexed: %d", count)

    logger.info("Indexing complete.")


if __name__ == "__main__":
    main()
