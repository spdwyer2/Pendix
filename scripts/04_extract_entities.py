#!/usr/bin/env python3
"""Extract person entities from transcripts and match against IMDB data.

Uses spaCy NER to find PERSON entities in episode transcripts, then
cross-references them with the IMDB name.basics dataset to identify
directors, actors, and cinematographers.
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import spacy
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def load_imdb_names(imdb_dir: Path) -> dict[str, dict]:
    """Load IMDB name.basics data into a lookup dict keyed by lowercase name.

    Args:
        imdb_dir: Directory containing the IMDB TSV files.

    Returns:
        Dict mapping lowercase name to list of IMDB person records.
    """
    names_file = imdb_dir / "name.basics.tsv"
    if not names_file.exists():
        raise FileNotFoundError(
            f"IMDB names file not found: {names_file}. Run 03_download_imdb.py first."
        )

    logger.info("Loading IMDB names from %s", names_file)
    name_lookup: dict[str, list[dict]] = defaultdict(list)

    with open(names_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            professions = row.get("primaryProfession", "")
            # Only include people with relevant professions
            relevant = {"actor", "actress", "director", "cinematographer", "writer", "producer"}
            person_professions = set(professions.split(",")) if professions != "\\N" else set()
            if not person_professions & relevant:
                continue

            name_lower = row.get("primaryName", "").lower().strip()
            if not name_lower:
                continue

            name_lookup[name_lower].append({
                "nconst": row.get("nconst", ""),
                "name": row.get("primaryName", ""),
                "birth_year": row.get("birthYear", ""),
                "professions": professions.split(",") if professions != "\\N" else [],
                "known_for_titles": (
                    row.get("knownForTitles", "").split(",")
                    if row.get("knownForTitles", "") != "\\N"
                    else []
                ),
            })

    logger.info("Loaded %d unique names from IMDB", len(name_lookup))
    return name_lookup


def categorize_person(professions: list[str]) -> str:
    """Determine primary category for a person based on their IMDB professions.

    Args:
        professions: List of profession strings from IMDB.

    Returns:
        One of: "director", "actor", "cinematographer", "other".
    """
    prof_set = set(professions)
    if "director" in prof_set:
        return "director"
    if "actor" in prof_set or "actress" in prof_set:
        return "actor"
    if "cinematographer" in prof_set:
        return "cinematographer"
    return "other"


def extract_entities_from_transcript(
    transcript: dict,
    nlp: spacy.language.Language,
    imdb_names: dict[str, list[dict]],
) -> dict:
    """Extract and match person entities from a single transcript.

    Args:
        transcript: Transcript dict with segments and full_text.
        nlp: Loaded spaCy model.
        imdb_names: IMDB name lookup dict.

    Returns:
        Dict with episode info and list of matched people.
    """
    # Track mentions: name -> list of timestamps
    mentions: dict[str, list[dict]] = defaultdict(list)

    # Process each segment to get timestamped mentions
    for segment in transcript.get("segments", []):
        doc = nlp(segment["text"])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if len(name.split()) >= 2:  # Require at least first + last name
                    mentions[name].append({
                        "start": segment["start"],
                        "end": segment["end"],
                    })

    # Match against IMDB
    matched_people = []
    seen_nconsts: set[str] = set()

    for name, timestamps in mentions.items():
        name_lower = name.lower().strip()
        imdb_matches = imdb_names.get(name_lower, [])

        if imdb_matches:
            # Use the first match (most common/notable person with that name)
            best = imdb_matches[0]
            if best["nconst"] in seen_nconsts:
                # Already matched this person via a different name variation
                # Add timestamps to existing entry
                for person in matched_people:
                    if person["nconst"] == best["nconst"]:
                        person["timestamps"].extend(timestamps)
                        break
                continue

            seen_nconsts.add(best["nconst"])
            matched_people.append({
                "name": best["name"],
                "name_as_mentioned": name,
                "nconst": best["nconst"],
                "professions": best["professions"],
                "category": categorize_person(best["professions"]),
                "known_for_titles": best["known_for_titles"],
                "mention_count": len(timestamps),
                "timestamps": timestamps,
            })

    # Sort by mention count (most mentioned first)
    matched_people.sort(key=lambda p: p["mention_count"], reverse=True)

    return {
        "episode_title": transcript.get("episode_title", ""),
        "episode_date": transcript.get("episode_date", ""),
        "audio_file": transcript.get("audio_file", ""),
        "total_people_found": len(matched_people),
        "people": matched_people,
    }


def main() -> None:
    """Extract entities from all transcripts and match against IMDB."""
    parser = argparse.ArgumentParser(
        description="Extract person entities from transcripts and match with IMDB."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process a specific transcript file (by filename in data/transcripts/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process files that already have entity files",
    )
    parser.add_argument(
        "--spacy-model",
        default=config.SPACY_MODEL,
        help=f"spaCy model to use (default: {config.SPACY_MODEL})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    config.ENTITIES_DIR.mkdir(parents=True, exist_ok=True)

    # Load spaCy model
    logger.info("Loading spaCy model: %s", args.spacy_model)
    try:
        nlp = spacy.load(args.spacy_model)
    except OSError:
        logger.error(
            "spaCy model '%s' not found. Install it with: python -m spacy download %s",
            args.spacy_model,
            args.spacy_model,
        )
        sys.exit(1)

    # Load IMDB names
    imdb_names = load_imdb_names(config.IMDB_DIR)

    # Find transcript files
    if args.file:
        transcript_path = config.TRANSCRIPTS_DIR / args.file
        if not transcript_path.exists():
            logger.error("Transcript file not found: %s", transcript_path)
            sys.exit(1)
        transcript_files = [transcript_path]
    else:
        transcript_files = sorted(config.TRANSCRIPTS_DIR.glob("*.json"))

    if not transcript_files:
        logger.error("No transcript files found in %s", config.TRANSCRIPTS_DIR)
        sys.exit(1)

    # Process each transcript
    all_mentions: list[dict] = []

    for tf in tqdm(transcript_files, desc="Extracting entities", unit="file"):
        entity_path = config.ENTITIES_DIR / tf.name

        if entity_path.exists() and not args.overwrite:
            logger.info("Entity file already exists, skipping: %s", tf.name)
            with open(entity_path, "r", encoding="utf-8") as f:
                all_mentions.append(json.load(f))
            continue

        with open(tf, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        result = extract_entities_from_transcript(transcript, nlp, imdb_names)

        # Save per-episode entity file
        with open(entity_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(
            "Saved entities for '%s': %d people found",
            result["episode_title"],
            result["total_people_found"],
        )

        all_mentions.append(result)

    # Save combined mentions file
    combined_path = config.ENTITIES_DIR / "all_mentions.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_mentions, f, indent=2, ensure_ascii=False)
    logger.info("Saved combined mentions: %s (%d episodes)", combined_path, len(all_mentions))


if __name__ == "__main__":
    main()
