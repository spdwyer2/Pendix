"""Central configuration for the Pendix data pipeline."""

import os
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Data Directories ─────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
IMDB_DIR = DATA_DIR / "imdb"
ENTITIES_DIR = DATA_DIR / "entities"

EPISODES_METADATA_PATH = DATA_DIR / "episodes_metadata.json"

# ── RSS Feed ──────────────────────────────────────────────────────────────────
RSS_FEED_URL = "https://feeds.megaphone.fm/the-rewatchables"
DEFAULT_EPISODE_LIMIT = 2

# ── Whisper Transcription ─────────────────────────────────────────────────────
# Model sizes: tiny, base, small, medium, large
# "medium" is recommended for M4 Pro (good speed/accuracy balance)
DEFAULT_WHISPER_MODEL = "medium"

# ── IMDB Dataset ──────────────────────────────────────────────────────────────
IMDB_BASE_URL = "https://datasets.imdbws.com/"
IMDB_FILES = [
    "name.basics.tsv.gz",
    "title.basics.tsv.gz",
    "title.crew.tsv.gz",
    "title.principals.tsv.gz",
]

# ── Elasticsearch ─────────────────────────────────────────────────────────────
# Supports both local ES and Elastic Cloud.  Set these env vars for cloud:
#   ELASTICSEARCH_URL  — e.g. https://my-deploy.es.us-east-1.aws.elastic.cloud:443
#   ES_API_KEY         — Base64-encoded API key from Elastic Cloud
ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
ES_API_KEY = os.environ.get("ES_API_KEY", "")
ES_INDEX_EPISODES = "pendix_episodes"
ES_INDEX_PEOPLE = "pendix_people"
ES_INDEX_MENTIONS = "pendix_mentions"

# ── spaCy ─────────────────────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
