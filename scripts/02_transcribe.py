#!/usr/bin/env python3
"""Transcribe podcast audio files using OpenAI Whisper (local).

Processes audio files from data/audio/ and outputs JSON transcripts with
segment-level timestamps to data/transcripts/.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import whisper
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def load_episode_metadata() -> dict[str, dict]:
    """Load episode metadata and return a dict keyed by slug.

    Returns:
        Dict mapping episode slug to its metadata.
    """
    if not config.EPISODES_METADATA_PATH.exists():
        logger.warning("No episodes_metadata.json found. Run 01_download_episodes.py first.")
        return {}

    with open(config.EPISODES_METADATA_PATH, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    return {ep["slug"]: ep for ep in episodes}


def find_audio_files(audio_dir: Path) -> list[Path]:
    """Find all audio files in the given directory.

    Args:
        audio_dir: Directory to search for audio files.

    Returns:
        Sorted list of audio file paths.
    """
    extensions = {".mp3", ".m4a", ".wav", ".ogg", ".flac"}
    files = [f for f in audio_dir.iterdir() if f.suffix.lower() in extensions]
    return sorted(files)


def transcribe_audio(
    audio_path: Path,
    model: whisper.Whisper,
    episode_meta: dict | None = None,
) -> dict:
    """Transcribe a single audio file using Whisper.

    Args:
        audio_path: Path to the audio file.
        model: Loaded Whisper model.
        episode_meta: Optional episode metadata to include in output.

    Returns:
        Dict containing transcript with segments and metadata.
    """
    logger.info("Transcribing: %s", audio_path.name)
    result = model.transcribe(str(audio_path), verbose=False)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "id": seg["id"],
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    transcript = {
        "episode_title": episode_meta.get("title", audio_path.stem) if episode_meta else audio_path.stem,
        "episode_date": episode_meta.get("date", "") if episode_meta else "",
        "audio_file": audio_path.name,
        "language": result.get("language", "en"),
        "full_text": result.get("text", "").strip(),
        "segments": segments,
    }

    return transcript


def save_transcript(transcript: dict, output_dir: Path) -> Path:
    """Save a transcript dict to a JSON file.

    Args:
        transcript: Transcript data dict.
        output_dir: Directory to write the JSON file.

    Returns:
        Path to the saved JSON file.
    """
    # Derive filename from audio filename
    audio_stem = Path(transcript["audio_file"]).stem
    output_path = output_dir / f"{audio_stem}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    logger.info("Saved transcript: %s", output_path.name)
    return output_path


def main() -> None:
    """Parse CLI args, load Whisper model, and transcribe audio files."""
    parser = argparse.ArgumentParser(
        description="Transcribe podcast audio using OpenAI Whisper (local)."
    )
    parser.add_argument(
        "--model",
        default=config.DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help=f"Whisper model size (default: {config.DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Transcribe a specific audio file (by filename in data/audio/)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-transcribe files that already have transcripts",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load episode metadata for enrichment
    episode_metadata = load_episode_metadata()

    # Find audio files to process
    if args.file:
        audio_path = config.AUDIO_DIR / args.file
        if not audio_path.exists():
            logger.error("Audio file not found: %s", audio_path)
            sys.exit(1)
        audio_files = [audio_path]
    else:
        audio_files = find_audio_files(config.AUDIO_DIR)

    if not audio_files:
        logger.error("No audio files found in %s", config.AUDIO_DIR)
        sys.exit(1)

    # Filter out already-transcribed files unless --overwrite
    if not args.overwrite:
        to_process = []
        for af in audio_files:
            transcript_path = config.TRANSCRIPTS_DIR / f"{af.stem}.json"
            if transcript_path.exists():
                logger.info("Transcript already exists, skipping: %s", af.name)
            else:
                to_process.append(af)
        audio_files = to_process

    if not audio_files:
        logger.info("All files already transcribed. Use --overwrite to redo.")
        return

    # Load Whisper model
    logger.info("Loading Whisper model: %s", args.model)
    model = whisper.load_model(args.model)
    logger.info("Model loaded successfully")

    # Transcribe each file
    for audio_path in tqdm(audio_files, desc="Transcribing", unit="file"):
        slug = audio_path.stem
        meta = episode_metadata.get(slug)
        transcript = transcribe_audio(audio_path, model, meta)
        save_transcript(transcript, config.TRANSCRIPTS_DIR)

    logger.info("Transcription complete. %d file(s) processed.", len(audio_files))


if __name__ == "__main__":
    main()
