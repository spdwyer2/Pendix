#!/usr/bin/env python3
"""Transcribe podcast audio files using OpenAI Whisper (local).

Processes audio files from data/audio/ and outputs JSON transcripts with
segment-level timestamps to data/transcripts/.

Optionally performs speaker diarization using pyannote.audio (--diarize flag)
to label each segment with a speaker ID (e.g. SPEAKER_00, SPEAKER_01).

When diarization is enabled, a speaker map template is written to
data/speaker_maps/{slug}.json containing sample utterances and voice
embeddings.  If a speaker library exists (built via manage_speakers.py),
speakers are auto-matched by voice similarity and names are pre-filled.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import whisper
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


# ── Episode metadata helpers ─────────────────────────────────────────────────

def load_episode_metadata() -> dict[str, dict]:
    """Load episode metadata and return a dict keyed by slug."""
    if not config.EPISODES_METADATA_PATH.exists():
        logger.warning("No episodes_metadata.json found. Run 01_download_episodes.py first.")
        return {}

    with open(config.EPISODES_METADATA_PATH, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    return {ep["slug"]: ep for ep in episodes}


def find_audio_files(audio_dir: Path) -> list[Path]:
    """Find all audio files in the given directory."""
    extensions = {".mp3", ".m4a", ".wav", ".ogg", ".flac"}
    files = [f for f in audio_dir.iterdir() if f.suffix.lower() in extensions]
    return sorted(files)


# ── Diarization ──────────────────────────────────────────────────────────────

def load_diarization_pipeline(hf_token: str):
    """Load pyannote diarization pipeline (once, reused across files).

    Returns:
        pyannote.audio.Pipeline instance.
    """
    from pyannote.audio import Pipeline

    logger.info("Loading diarization pipeline: %s", config.DEFAULT_DIARIZATION_MODEL)
    pipeline = Pipeline.from_pretrained(
        config.DEFAULT_DIARIZATION_MODEL,
        use_auth_token=hf_token,
    )
    return pipeline


def load_embedding_model(hf_token: str):
    """Load pyannote speaker embedding model (once, reused across files).

    Returns:
        pyannote.audio.Inference instance for extracting embeddings.
    """
    from pyannote.audio import Inference

    logger.info("Loading embedding model: %s", config.DEFAULT_EMBEDDING_MODEL)
    return Inference(config.DEFAULT_EMBEDDING_MODEL, use_auth_token=hf_token)


def run_diarization(audio_path: Path, pipeline) -> list[dict]:
    """Run speaker diarization on an audio file.

    Args:
        audio_path: Path to the audio file.
        pipeline: Loaded pyannote diarization pipeline.

    Returns:
        List of diarization turns with 'start', 'end', and 'speaker'.
    """
    logger.info("Running speaker diarization on: %s", audio_path.name)
    diarization = pipeline(str(audio_path))

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
        })

    logger.info(
        "Diarization complete: %d turns, %d speakers",
        len(turns),
        len({t["speaker"] for t in turns}),
    )
    return turns


# ── Speaker embeddings ───────────────────────────────────────────────────────

def extract_speaker_embeddings(
    audio_path: Path,
    diarization_turns: list[dict],
    embedding_model,
    max_segments_per_speaker: int = 10,
) -> dict[str, list[float]]:
    """Extract an average voice embedding per speaker from diarization turns.

    For each speaker, samples up to *max_segments_per_speaker* of their
    longest turns, extracts an embedding for each, and averages them.

    Args:
        audio_path: Path to the audio file.
        diarization_turns: Diarization turns from run_diarization().
        embedding_model: Loaded pyannote Inference embedding model.
        max_segments_per_speaker: Max segments to sample per speaker.

    Returns:
        Dict mapping speaker label to embedding (list of floats).
    """
    from pyannote.core import Segment

    # Group turns by speaker, sorted by duration (longest first)
    speaker_turns: dict[str, list[dict]] = {}
    for turn in diarization_turns:
        speaker_turns.setdefault(turn["speaker"], []).append(turn)

    embeddings: dict[str, list[float]] = {}
    for speaker, turns in speaker_turns.items():
        # Sort by duration descending, take longest segments for best embeddings
        turns_sorted = sorted(turns, key=lambda t: t["end"] - t["start"], reverse=True)
        sample = turns_sorted[:max_segments_per_speaker]

        # Filter out very short segments (< 1 second) — poor embedding quality
        sample = [t for t in sample if (t["end"] - t["start"]) >= 1.0]
        if not sample:
            sample = turns_sorted[:1]  # fallback to whatever we have

        segment_embeddings = []
        for turn in sample:
            try:
                emb = embedding_model.crop(
                    str(audio_path),
                    Segment(turn["start"], turn["end"]),
                )
                segment_embeddings.append(emb)
            except Exception:
                logger.debug("Failed to extract embedding for %s [%.1f-%.1f]",
                             speaker, turn["start"], turn["end"])
                continue

        if segment_embeddings:
            avg = np.mean(segment_embeddings, axis=0)
            embeddings[speaker] = avg.tolist()
        else:
            logger.warning("No embeddings extracted for %s", speaker)

    return embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


# ── Speaker library ──────────────────────────────────────────────────────────

def load_speaker_library() -> dict[str, dict]:
    """Load the speaker voice library from disk.

    Returns:
        Dict mapping person name to {"embedding": [...], "confirmed_episodes": N}.
        Empty dict if no library exists.
    """
    if not config.SPEAKER_LIBRARY_PATH.exists():
        return {}

    with open(config.SPEAKER_LIBRARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def match_speakers_to_library(
    speaker_embeddings: dict[str, list[float]],
    library: dict[str, dict],
    threshold: float | None = None,
) -> dict[str, str]:
    """Match speaker embeddings against the voice library.

    Args:
        speaker_embeddings: Dict mapping speaker label to embedding.
        library: Voice library (from load_speaker_library).
        threshold: Cosine similarity threshold for a match.

    Returns:
        Dict mapping speaker label to matched person name.
        Only includes speakers that matched above threshold.
    """
    if threshold is None:
        threshold = config.SPEAKER_MATCH_THRESHOLD

    if not library:
        return {}

    matches: dict[str, str] = {}
    used_names: set[str] = set()

    # Score all (speaker, library_name) pairs
    scores: list[tuple[str, str, float]] = []
    for speaker_label, emb in speaker_embeddings.items():
        for name, entry in library.items():
            sim = cosine_similarity(emb, entry["embedding"])
            scores.append((speaker_label, name, sim))

    # Greedy assignment: best score first, no double-assignments
    scores.sort(key=lambda x: x[2], reverse=True)
    for speaker_label, name, sim in scores:
        if speaker_label in matches or name in used_names:
            continue
        if sim >= threshold:
            matches[speaker_label] = name
            used_names.add(name)
            logger.info(
                "Auto-matched %s → %s (similarity: %.3f)",
                speaker_label, name, sim,
            )

    return matches


# ── Speaker map generation ───────────────────────────────────────────────────

def generate_speaker_map(
    transcript: dict,
    diarization_turns: list[dict],
    speaker_embeddings: dict[str, list[float]],
    auto_matches: dict[str, str],
    episode_meta: dict | None = None,
) -> dict:
    """Generate a speaker map template for an episode.

    Includes sample utterances, speaking time, embeddings, and any
    auto-matched names from the voice library.

    Args:
        transcript: The transcript dict (with diarized segments).
        diarization_turns: Raw diarization turns.
        speaker_embeddings: Embeddings per speaker label.
        auto_matches: Auto-matched names from the voice library.
        episode_meta: Episode metadata (for description/candidates).

    Returns:
        Speaker map dict.
    """
    # Collect per-speaker stats from transcript segments
    speaker_segments: dict[str, list[dict]] = {}
    for seg in transcript.get("segments", []):
        speaker = seg.get("speaker", "UNKNOWN")
        speaker_segments.setdefault(speaker, []).append(seg)

    # Compute speaking time from diarization turns
    speaker_time: dict[str, float] = {}
    for turn in diarization_turns:
        sp = turn["speaker"]
        speaker_time[sp] = speaker_time.get(sp, 0.0) + (turn["end"] - turn["start"])

    speakers_info: dict[str, dict] = {}
    for speaker_label in sorted(speaker_segments.keys()):
        segs = speaker_segments[speaker_label]
        # Pick sample utterances: longest segments with actual content
        samples = sorted(segs, key=lambda s: len(s.get("text", "")), reverse=True)
        sample_texts = [s["text"] for s in samples[:5] if s.get("text")]

        speakers_info[speaker_label] = {
            "name": auto_matches.get(speaker_label, ""),
            "auto_matched": speaker_label in auto_matches,
            "total_speaking_time_seconds": round(speaker_time.get(speaker_label, 0.0), 1),
            "segment_count": len(segs),
            "sample_utterances": sample_texts,
            "embedding": speaker_embeddings.get(speaker_label, []),
        }

    speaker_map = {
        "episode_title": transcript.get("episode_title", ""),
        "episode_date": transcript.get("episode_date", ""),
        "audio_file": transcript.get("audio_file", ""),
        "description": episode_meta.get("description", "") if episode_meta else "",
        "speakers": speakers_info,
    }

    return speaker_map


def save_speaker_map(speaker_map: dict, slug: str) -> Path:
    """Save a speaker map to data/speaker_maps/{slug}.json."""
    config.SPEAKER_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.SPEAKER_MAPS_DIR / f"{slug}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, indent=2, ensure_ascii=False)

    logger.info("Saved speaker map: %s", output_path.name)
    return output_path


# ── Speaker assignment ───────────────────────────────────────────────────────

def assign_speakers(
    segments: list[dict],
    diarization_turns: list[dict],
    name_map: dict[str, str] | None = None,
) -> list[dict]:
    """Assign speaker labels to Whisper segments based on diarization overlap.

    For each Whisper segment, finds the diarization turn with the greatest
    time overlap and assigns that speaker label.  If *name_map* is provided,
    maps generic labels (SPEAKER_00) to real names.

    Args:
        segments: Whisper transcript segments.
        diarization_turns: Diarization turns.
        name_map: Optional mapping from speaker labels to real names.

    Returns:
        Segments with 'speaker' field added.
    """
    for segment in segments:
        seg_start = segment["start"]
        seg_end = segment["end"]

        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for turn in diarization_turns:
            overlap_start = max(seg_start, turn["start"])
            overlap_end = min(seg_end, turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        # Apply name mapping if available
        if name_map and best_speaker in name_map:
            segment["speaker"] = name_map[best_speaker]
        else:
            segment["speaker"] = best_speaker

    return segments


# ── Transcription ────────────────────────────────────────────────────────────

def transcribe_audio(
    audio_path: Path,
    model: Any,
    episode_meta: Optional[dict] = None,
    diarize: bool = False,
    diarization_pipeline=None,
    embedding_model=None,
    speaker_library: dict[str, dict] | None = None,
) -> dict:
    """Transcribe a single audio file, optionally with speaker diarization.

    When diarization is enabled, also extracts voice embeddings, matches
    against the speaker library, and generates a speaker map template.

    Returns:
        Transcript dict.
    """
    diarization_turns: list[dict] = []
    speaker_embeddings: dict[str, list[float]] = {}
    auto_matches: dict[str, str] = {}

    if diarize and diarization_pipeline is not None:
        diarization_turns = run_diarization(audio_path, diarization_pipeline)

        # Extract voice embeddings per speaker
        if embedding_model is not None and diarization_turns:
            speaker_embeddings = extract_speaker_embeddings(
                audio_path, diarization_turns, embedding_model,
            )

            # Auto-match against library
            if speaker_library:
                auto_matches = match_speakers_to_library(
                    speaker_embeddings, speaker_library,
                )

    # Run Whisper transcription
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

    # Merge speaker labels into segments
    if diarize and diarization_turns:
        segments = assign_speakers(segments, diarization_turns, auto_matches)

    transcript = {
        "episode_title": episode_meta.get("title", audio_path.stem) if episode_meta else audio_path.stem,
        "episode_date": episode_meta.get("date", "") if episode_meta else "",
        "audio_file": audio_path.name,
        "language": result.get("language", "en"),
        "diarized": diarize and bool(diarization_turns),
        "full_text": result.get("text", "").strip(),
        "segments": segments,
    }

    # Generate speaker map template
    if diarize and diarization_turns:
        slug = audio_path.stem
        speaker_map = generate_speaker_map(
            transcript, diarization_turns, speaker_embeddings,
            auto_matches, episode_meta,
        )
        save_speaker_map(speaker_map, slug)

    return transcript


def save_transcript(transcript: dict, output_dir: Path) -> Path:
    """Save a transcript dict to a JSON file."""
    audio_stem = Path(transcript["audio_file"]).stem
    output_path = output_dir / f"{audio_stem}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    logger.info("Saved transcript: %s", output_path.name)
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI args, load models, and transcribe audio files."""
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
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization via pyannote.audio (requires HF_AUTH_TOKEN)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    # Validate diarization prerequisites
    hf_token = config.HF_AUTH_TOKEN
    if args.diarize and not hf_token:
        logger.error(
            "Speaker diarization requires a HuggingFace auth token. "
            "Set the HF_AUTH_TOKEN environment variable. "
            "Get a token at https://huggingface.co/settings/tokens and "
            "accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
        sys.exit(1)

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

    # Load diarization models (once for all files)
    diarization_pipeline = None
    embedding_model = None
    speaker_library: dict[str, dict] = {}

    if args.diarize:
        diarization_pipeline = load_diarization_pipeline(hf_token)
        embedding_model = load_embedding_model(hf_token)
        speaker_library = load_speaker_library()
        if speaker_library:
            logger.info(
                "Loaded speaker library with %d known voices",
                len(speaker_library),
            )

    # Transcribe each file
    for audio_path in tqdm(audio_files, desc="Transcribing", unit="file"):
        slug = audio_path.stem
        meta = episode_metadata.get(slug)
        transcript = transcribe_audio(
            audio_path, model, meta,
            diarize=args.diarize,
            diarization_pipeline=diarization_pipeline,
            embedding_model=embedding_model,
            speaker_library=speaker_library,
        )
        save_transcript(transcript, config.TRANSCRIPTS_DIR)

    logger.info("Transcription complete. %d file(s) processed.", len(audio_files))


if __name__ == "__main__":
    main()
