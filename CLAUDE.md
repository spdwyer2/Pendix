# Pendix — Claude Code Session Guide

## What is Pendix?

Pendix is a movie podcast analytics site. It processes podcast transcripts to surface statistics about directors, actors, and cinematographers mentioned across episodes. The first podcast is **The Rewatchables** from The Ringer.

## Project Structure

- `config.py` — All paths, URLs, index names, and defaults. Import this in every script.
- `scripts/` — Numbered pipeline scripts (01–05) plus `run_pipeline.py` orchestrator.
- `data/` — Gitignored. Contains audio, transcripts, IMDB TSVs, and extracted entities. Created by the pipeline scripts.
- `elasticsearch/` — Docker Compose for ES+Kibana and index mapping JSON files.
- `web/` — Website (Flask or FastAPI, to be built).

## Conventions

- Python 3.11+. Use type hints.
- Use `logging` module, not `print()`.
- All scripts use `argparse` and have `if __name__ == "__main__"` blocks.
- All paths are defined in `config.py` — never hardcode paths in scripts.
- Dependencies are in `requirements.txt` and `pyproject.toml`.

## Pipeline Steps

Run all: `python scripts/run_pipeline.py`

| Step | Script | What it does |
|------|--------|-------------|
| download | `01_download_episodes.py` | Fetches RSS feed, downloads audio to `data/audio/` |
| transcribe | `02_transcribe.py` | Runs Whisper on audio, outputs JSON transcripts to `data/transcripts/` |
| imdb | `03_download_imdb.py` | Downloads IMDB public datasets to `data/imdb/` |
| entities | `04_extract_entities.py` | Uses spaCy NER to find people in transcripts, matches against IMDB |
| index | `05_index_elasticsearch.py` | Indexes transcripts and people into Elasticsearch |

## Key Data Formats

### Episode metadata (`data/episodes_metadata.json`)
```json
[{"title": "...", "date": "YYYY-MM-DD", "description": "...", "audio_url": "...", "slug": "..."}]
```

### Transcript (`data/transcripts/{slug}.json`)
```json
{"episode_title": "...", "episode_date": "...", "full_text": "...", "segments": [{"id": 0, "start": 0.0, "end": 5.2, "text": "..."}]}
```

### Entity extraction (`data/entities/{slug}.json`)
```json
{"episode_title": "...", "people": [{"name": "...", "nconst": "nm...", "category": "director|actor|cinematographer|other", "mention_count": 5, "timestamps": [...]}]}
```

## Elasticsearch

Start: `cd elasticsearch && docker compose up -d`

Indices: `pendix_episodes`, `pendix_people` (defined in `config.py`).
Mappings: `elasticsearch/mappings/*.json`.

## Running Locally

This project runs on a MacBook Pro M4 Pro. The data pipeline is not meant to run in cloud/CI environments — audio downloads and Whisper transcription require local execution.

## What Needs Building Next

- **Task 2**: Web frontend (Flask or FastAPI) that queries Elasticsearch and displays stats.
- **Task 3**: Polish, deploy, or extend with more podcasts.
