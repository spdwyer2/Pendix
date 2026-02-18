# Pendix

Movie podcast analytics — statistics about directors, actors, and cinematographers mentioned on podcasts.

Starting with **The Rewatchables** from The Ringer network, Pendix helps movie lovers go deeper on the filmmakers and performers discussed across podcast episodes.

## How It Works

Pendix runs a local data pipeline that:

1. **Downloads** podcast episodes from an RSS feed
2. **Transcribes** audio using OpenAI Whisper (locally on your machine)
3. **Downloads** IMDB public datasets for cross-referencing
4. **Extracts** person entities (directors, actors, cinematographers) from transcripts using spaCy NER and matches them against IMDB
5. **Indexes** everything into Elasticsearch for search and analytics

## Prerequisites

- **Python 3.11+**
- **Docker** and **Docker Compose** (for Elasticsearch + Kibana)
- **ffmpeg** (required by Whisper for audio processing)

### macOS (Homebrew)

```bash
brew install python@3.11 ffmpeg docker docker-compose
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url> Pendix
cd Pendix
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 3. Start Elasticsearch and Kibana

```bash
cd elasticsearch
docker compose up -d
cd ..
```

Elasticsearch will be available at `http://localhost:9200` and Kibana at `http://localhost:5601`.

## Usage

### Run the full pipeline

```bash
python scripts/run_pipeline.py
```

By default this downloads the 2 most recent episodes. The pipeline runs these steps in order: download → transcribe → imdb → entities → index.

### Run individual steps

```bash
# Download episodes (default: 2 most recent)
python scripts/01_download_episodes.py

# Download more episodes
python scripts/01_download_episodes.py --num-episodes 10

# Download all episodes
python scripts/01_download_episodes.py --all

# Only save metadata (skip audio download)
python scripts/01_download_episodes.py --metadata-only

# Transcribe audio files
python scripts/02_transcribe.py

# Use a different Whisper model (tiny/base/small/medium/large)
python scripts/02_transcribe.py --model small

# Download IMDB datasets
python scripts/03_download_imdb.py

# Extract entities from transcripts
python scripts/04_extract_entities.py

# Index into Elasticsearch
python scripts/05_index_elasticsearch.py
```

### Run pipeline from a specific step

```bash
# Resume from the transcription step onward
python scripts/run_pipeline.py --from transcribe

# Run only the entity extraction step
python scripts/run_pipeline.py --step entities

# Pass extra args to a single step
python scripts/run_pipeline.py --step download -- --num-episodes 5
```

## Speaker Diarization & Identification

Pendix can identify **who is speaking** each line of a transcript using [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization and a voice embedding library for recognizing recurring hosts across episodes.

### Prerequisites

1. Get a [HuggingFace token](https://huggingface.co/settings/tokens)
2. Accept the model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/embedding](https://huggingface.co/pyannote/embedding)
3. Set the environment variable:
   ```bash
   export HF_AUTH_TOKEN=hf_your_token_here
   ```

### First episode (bootstrap)

```bash
# 1. Transcribe with diarization — generates transcripts + speaker maps
python scripts/02_transcribe.py --diarize

# 2. Check the speaker map — it shows sample utterances per speaker
#    so you can tell who is who
cat data/speaker_maps/heat-1995.json
```

The speaker map looks like this:

```json
{
  "speakers": {
    "SPEAKER_00": {
      "name": "",
      "total_speaking_time_seconds": 1842.3,
      "sample_utterances": [
        "Welcome back to The Rewatchables...",
        "I think what makes this movie so rewatchable..."
      ],
      "embedding": [0.12, -0.34, "..."]
    },
    "SPEAKER_01": { "..." : "..." }
  },
  "description": "Bill Simmons, Chris Ryan, and Sean Fennessey..."
}
```

```bash
# 3. Edit the speaker map — fill in the "name" field for each speaker
#    (the episode description and sample utterances help you identify them)

# 4. Confirm it — saves voice embeddings to the library
python scripts/manage_speakers.py confirm heat-1995

# 5. Apply names back into the transcript JSON
python scripts/manage_speakers.py apply
```

### Future episodes (auto-matched)

Once voices are in the library, they're recognized automatically:

```bash
python scripts/02_transcribe.py --diarize --file new-episode.mp3
# → "Auto-matched SPEAKER_00 → Bill Simmons (similarity: 0.891)"
```

The voice library improves with each confirmed episode (embeddings are averaged over time).

### Speaker management commands

```bash
# List all known voices in the library
python scripts/manage_speakers.py list

# Show which speaker maps are confirmed vs pending
python scripts/manage_speakers.py status

# Apply names even from unconfirmed maps
python scripts/manage_speakers.py apply --include-unconfirmed
```

## Project Structure

```
Pendix/
├── config.py                     # Central configuration (paths, URLs, defaults)
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project metadata and build config
│
├── scripts/
│   ├── 01_download_episodes.py   # Download podcast audio from RSS
│   ├── 02_transcribe.py          # Transcribe audio with Whisper + diarization
│   ├── 03_download_imdb.py       # Download IMDB public datasets
│   ├── 04_extract_entities.py    # Extract directors/actors/cinematographers
│   ├── 05_index_elasticsearch.py # Index into Elasticsearch
│   ├── manage_speakers.py        # Confirm speaker maps & manage voice library
│   └── run_pipeline.py           # Orchestrate all steps
│
├── data/                         # .gitignored — local data storage
│   ├── audio/                    # Downloaded podcast audio files
│   ├── transcripts/              # JSON transcripts with timestamps
│   ├── imdb/                     # IMDB TSV dataset files
│   ├── entities/                 # Extracted entity JSON files
│   ├── speaker_maps/             # Per-episode speaker mapping templates
│   └── speaker_library.json      # Known voice embeddings
│
├── elasticsearch/
│   ├── mappings/
│   │   ├── episodes.json         # ES mapping for episode transcripts
│   │   └── imdb_people.json      # ES mapping for people data
│   └── docker-compose.yml        # Elasticsearch + Kibana setup
│
├── web/                          # Website (to be built)
├── .gitignore
├── CLAUDE.md                     # Instructions for Claude Code sessions
└── README.md
```

## Data Files

All data files are stored in `data/` (gitignored). Key outputs:

- `data/episodes_metadata.json` — Episode titles, dates, descriptions, audio URLs
- `data/transcripts/*.json` — Full transcripts with segment-level timestamps
- `data/entities/*.json` — Per-episode matched people with mention counts and timestamps
- `data/entities/all_mentions.json` — Combined cross-episode entity data

## Whisper Model Selection

| Model  | Size  | Speed (M4 Pro) | Accuracy | Recommended For        |
|--------|-------|-----------------|----------|------------------------|
| tiny   | 39M   | Fastest         | Lower    | Quick testing          |
| base   | 74M   | Fast            | Fair     | Fast iteration         |
| small  | 244M  | Moderate        | Good     | Reasonable quality     |
| medium | 769M  | Slower          | Better   | **Default — best balance** |
| large  | 1.5G  | Slowest         | Best     | Maximum accuracy       |

The `medium` model is the default and recommended for M4 Pro machines.

## Elasticsearch Indices

| Index            | Description                                    |
|------------------|------------------------------------------------|
| `pendix_episodes`| Episode transcripts with full text and segments|
| `pendix_people`  | IMDB people with episode mention counts        |
