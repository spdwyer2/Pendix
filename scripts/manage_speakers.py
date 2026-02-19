#!/usr/bin/env python3
"""Manage the speaker voice library and apply speaker names to transcripts.

Subcommands
-----------
confirm   Confirm a speaker map (after you've filled in names), updating the
          voice library with the new embeddings.
apply     Apply confirmed speaker names from speaker maps back into transcript
          JSON files, replacing generic labels (SPEAKER_00) with real names.
list      List all known voices in the speaker library.
status    Show which speaker maps are confirmed vs pending.

Workflow
--------
1. Run ``02_transcribe.py --diarize`` → generates speaker maps in data/speaker_maps/
2. Edit the speaker map JSON: fill in the ``"name"`` field for each speaker.
3. Run ``manage_speakers.py confirm <slug>`` → saves voice embeddings to library.
4. Run ``manage_speakers.py apply`` → updates transcript segments with real names.
5. Future episodes auto-match known voices during diarization.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


# ── Library I/O ──────────────────────────────────────────────────────────────

def load_speaker_library() -> dict[str, dict]:
    """Load the speaker voice library."""
    if not config.SPEAKER_LIBRARY_PATH.exists():
        return {}
    with open(config.SPEAKER_LIBRARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_speaker_library(library: dict[str, dict]) -> None:
    """Save the speaker voice library."""
    config.SPEAKER_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SPEAKER_LIBRARY_PATH, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)
    logger.info("Saved speaker library (%d voices)", len(library))


# ── confirm ──────────────────────────────────────────────────────────────────

def cmd_confirm(args: argparse.Namespace) -> None:
    """Confirm a speaker map and update the voice library."""
    slug = args.slug
    map_path = config.SPEAKER_MAPS_DIR / f"{slug}.json"

    if not map_path.exists():
        logger.error("Speaker map not found: %s", map_path)
        sys.exit(1)

    with open(map_path, "r", encoding="utf-8") as f:
        speaker_map = json.load(f)

    speakers = speaker_map.get("speakers", {})

    # Validate that all speakers have names
    unnamed = [label for label, info in speakers.items() if not info.get("name")]
    if unnamed:
        logger.error(
            "These speakers still need names: %s\n"
            "Edit %s and fill in the 'name' field for each speaker.",
            ", ".join(unnamed),
            map_path,
        )
        sys.exit(1)

    # Load existing library
    library = load_speaker_library()

    # Update library with new/updated voice embeddings
    for label, info in speakers.items():
        name = info["name"]
        embedding = info.get("embedding", [])

        if not embedding:
            logger.warning("No embedding for %s (%s) — skipping library update", label, name)
            continue

        if name in library:
            # Running average: blend the new embedding with the existing one
            existing_emb = np.array(library[name]["embedding"])
            new_emb = np.array(embedding)
            n = library[name].get("confirmed_episodes", 1)
            blended = ((existing_emb * n) + new_emb) / (n + 1)
            library[name]["embedding"] = blended.tolist()
            library[name]["confirmed_episodes"] = n + 1
            logger.info("Updated voice for '%s' (%d episodes)", name, n + 1)
        else:
            library[name] = {
                "embedding": embedding,
                "confirmed_episodes": 1,
            }
            logger.info("Added new voice: '%s'", name)

    save_speaker_library(library)

    # Mark the speaker map as confirmed
    speaker_map["confirmed"] = True
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, indent=2, ensure_ascii=False)

    logger.info("Confirmed speaker map for '%s'", speaker_map.get("episode_title", slug))


# ── apply ────────────────────────────────────────────────────────────────────

def cmd_apply(args: argparse.Namespace) -> None:
    """Apply confirmed speaker maps to transcript files."""
    if not config.SPEAKER_MAPS_DIR.exists():
        logger.error("No speaker maps directory found. Run diarization first.")
        sys.exit(1)

    map_files = sorted(config.SPEAKER_MAPS_DIR.glob("*.json"))
    if not map_files:
        logger.error("No speaker map files found in %s", config.SPEAKER_MAPS_DIR)
        sys.exit(1)

    applied = 0
    for map_path in map_files:
        with open(map_path, "r", encoding="utf-8") as f:
            speaker_map = json.load(f)

        if not speaker_map.get("confirmed") and not args.include_unconfirmed:
            logger.info("Skipping unconfirmed map: %s (use --include-unconfirmed to override)",
                        map_path.name)
            continue

        # Build name mapping: label → name
        name_map: dict[str, str] = {}
        for label, info in speaker_map.get("speakers", {}).items():
            name = info.get("name", "")
            if name:
                name_map[label] = name

        if not name_map:
            continue

        # Find and update the corresponding transcript
        slug = map_path.stem
        transcript_path = config.TRANSCRIPTS_DIR / f"{slug}.json"

        if not transcript_path.exists():
            logger.warning("Transcript not found for %s — skipping", slug)
            continue

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        # Update speaker labels in segments
        updated = False
        for segment in transcript.get("segments", []):
            old_speaker = segment.get("speaker", "")
            if old_speaker in name_map:
                segment["speaker"] = name_map[old_speaker]
                updated = True

        if updated:
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)
            logger.info("Applied speaker names to: %s", transcript_path.name)
            applied += 1

    logger.info("Applied speaker names to %d transcript(s)", applied)


# ── list ─────────────────────────────────────────────────────────────────────

def cmd_list(args: argparse.Namespace) -> None:
    """List all known voices in the speaker library."""
    library = load_speaker_library()

    if not library:
        logger.info("Speaker library is empty. Confirm some speaker maps first.")
        return

    print(f"\nSpeaker Library ({len(library)} voices):")
    print("-" * 50)
    for name, info in sorted(library.items()):
        episodes = info.get("confirmed_episodes", 0)
        emb_dim = len(info.get("embedding", []))
        print(f"  {name:<30} {episodes} episode(s)  [{emb_dim}-dim embedding]")
    print()


# ── status ───────────────────────────────────────────────────────────────────

def cmd_status(args: argparse.Namespace) -> None:
    """Show which speaker maps are confirmed vs pending."""
    if not config.SPEAKER_MAPS_DIR.exists():
        logger.info("No speaker maps directory found.")
        return

    map_files = sorted(config.SPEAKER_MAPS_DIR.glob("*.json"))
    if not map_files:
        logger.info("No speaker map files found.")
        return

    confirmed = []
    pending = []

    for map_path in map_files:
        with open(map_path, "r", encoding="utf-8") as f:
            speaker_map = json.load(f)

        title = speaker_map.get("episode_title", map_path.stem)
        speakers = speaker_map.get("speakers", {})
        named = sum(1 for s in speakers.values() if s.get("name"))
        total = len(speakers)

        entry = f"  {title:<50} {named}/{total} speakers named"
        if speaker_map.get("confirmed"):
            confirmed.append(entry)
        else:
            pending.append(entry)

    print(f"\nSpeaker Maps Status:")
    print("=" * 70)

    if confirmed:
        print(f"\nConfirmed ({len(confirmed)}):")
        for e in confirmed:
            print(e)

    if pending:
        print(f"\nPending ({len(pending)}):")
        for e in pending:
            print(e)

    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI args and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        description="Manage speaker voice library and apply names to transcripts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # confirm
    p_confirm = subparsers.add_parser(
        "confirm",
        help="Confirm a speaker map and update the voice library",
    )
    p_confirm.add_argument("slug", help="Episode slug (filename without .json)")

    # apply
    p_apply = subparsers.add_parser(
        "apply",
        help="Apply confirmed speaker names to transcript files",
    )
    p_apply.add_argument(
        "--include-unconfirmed",
        action="store_true",
        help="Also apply names from unconfirmed speaker maps",
    )

    # list
    subparsers.add_parser("list", help="List all known voices in the library")

    # status
    subparsers.add_parser("status", help="Show speaker map confirmation status")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    commands = {
        "confirm": cmd_confirm,
        "apply": cmd_apply,
        "list": cmd_list,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
