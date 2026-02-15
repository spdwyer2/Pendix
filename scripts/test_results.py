#!/usr/bin/env python3
"""Query Elasticsearch and display pipeline results as a pandas DataFrame.

Development/debugging tool for inspecting entity extraction quality before
scaling up or building the website. Outputs one row per person-mention per
episode with relevant metadata from the episodes and people indices.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from elasticsearch import Elasticsearch
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def get_es_client(url: str) -> Elasticsearch:
    """Create and verify an Elasticsearch connection.

    Args:
        url: Elasticsearch URL.

    Returns:
        Connected Elasticsearch client.

    Raises:
        SystemExit: If Elasticsearch is unreachable or the required indices
            don't exist.
    """
    try:
        es = Elasticsearch(url)
        if not es.ping():
            raise ConnectionError
    except Exception:
        logger.error(
            "Cannot connect to Elasticsearch at %s.\n"
            "  1. Start Elasticsearch:  cd elasticsearch && docker compose up -d\n"
            "  2. Index data:           python scripts/05_index_elasticsearch.py",
            url,
        )
        sys.exit(1)

    for idx in (config.ES_INDEX_EPISODES, config.ES_INDEX_PEOPLE):
        if not es.indices.exists(index=idx):
            logger.error(
                "Index '%s' does not exist. Run 05_index_elasticsearch.py first.",
                idx,
            )
            sys.exit(1)

    doc_count = int(es.cat.count(index=config.ES_INDEX_EPISODES, format="json")[0]["count"])
    if doc_count == 0:
        logger.error(
            "Index '%s' is empty. Run the full pipeline (scripts/run_pipeline.py) first.",
            config.ES_INDEX_EPISODES,
        )
        sys.exit(1)

    return es


def fetch_episodes(
    es: Elasticsearch, episodes_arg: str
) -> list[dict]:
    """Fetch episode documents from Elasticsearch based on the CLI filter.

    Args:
        es: Elasticsearch client.
        episodes_arg: One of "latest", "all", or a comma-separated list of
            slugs / dates.

    Returns:
        List of episode source dicts.
    """
    if episodes_arg == "all":
        body: dict = {"query": {"match_all": {}}, "sort": [{"episode_date": "desc"}], "size": 10000}
    elif episodes_arg == "latest":
        body = {"query": {"match_all": {}}, "sort": [{"episode_date": "desc"}], "size": 2}
    else:
        # Comma-separated slugs or dates
        tokens = [t.strip() for t in episodes_arg.split(",") if t.strip()]
        clauses: list[dict] = []
        for token in tokens:
            # Dates look like YYYY-MM-DD
            if len(token) == 10 and token[4] == "-" and token[7] == "-":
                clauses.append({"term": {"episode_date": token}})
            else:
                clauses.append({"term": {"slug": token}})
        body = {
            "query": {"bool": {"should": clauses, "minimum_should_match": 1}},
            "sort": [{"episode_date": "desc"}],
            "size": 10000,
        }

    resp = es.search(index=config.ES_INDEX_EPISODES, body=body)
    episodes = [hit["_source"] for hit in resp["hits"]["hits"]]

    if not episodes:
        logger.warning("No episodes matched the filter '%s'.", episodes_arg)

    return episodes


def fetch_people(es: Elasticsearch) -> dict[str, dict]:
    """Fetch all people from the people index, keyed by nconst.

    Args:
        es: Elasticsearch client.

    Returns:
        Dict mapping nconst to the person source document.
    """
    people: dict[str, dict] = {}
    body: dict = {"query": {"match_all": {}}, "size": 1000}

    resp = es.search(index=config.ES_INDEX_PEOPLE, body=body, scroll="2m")
    scroll_id = resp.get("_scroll_id")
    hits = resp["hits"]["hits"]

    while hits:
        for hit in hits:
            src = hit["_source"]
            people[src["nconst"]] = src
        resp = es.scroll(scroll_id=scroll_id, scroll="2m")
        hits = resp["hits"]["hits"]

    if scroll_id:
        es.clear_scroll(scroll_id=scroll_id)

    return people


def resolve_title_ids(es: Elasticsearch, title_ids: list[str]) -> dict[str, str]:
    """Resolve IMDB title IDs to human-readable titles via the episodes index.

    This is a best-effort lookup — it only resolves titles that happen to
    exist in our episodes index. For a richer lookup you'd query IMDB's
    title.basics data directly, but this is sufficient for a dev tool.

    Args:
        es: Elasticsearch client.
        title_ids: List of IMDB tconst strings (e.g. "tt0443706").

    Returns:
        Dict mapping tconst -> title string (only for resolved IDs).
    """
    # The episodes index doesn't store tconst, so we can't resolve here.
    # Return the raw IDs — still useful for sanity-checking.
    return {tid: tid for tid in title_ids}


def extract_sample_quote(
    episode: dict, person_name: str, timestamp: float | None
) -> str:
    """Pull a short transcript excerpt around a person's mention.

    Searches episode segments near the given timestamp for the person's name
    and returns the matching segment text (truncated to 120 chars).

    Args:
        episode: Episode source dict with segments.
        person_name: The person's name to search for.
        timestamp: Start time (seconds) of the mention.

    Returns:
        A short quote string, or "" if nothing found.
    """
    segments = episode.get("segments", [])
    if not segments:
        return ""

    # If we have a timestamp, find the closest segment
    if timestamp is not None:
        best_seg = None
        best_dist = float("inf")
        for seg in segments:
            dist = abs(seg.get("start", 0) - timestamp)
            if dist < best_dist:
                best_dist = dist
                best_seg = seg
        if best_seg:
            text = best_seg.get("text", "").strip()
            if len(text) > 120:
                text = text[:117] + "..."
            return text

    # Fallback: search for the name in any segment
    last_name = person_name.split()[-1].lower() if person_name else ""
    for seg in segments:
        if last_name and last_name in seg.get("text", "").lower():
            text = seg["text"].strip()
            if len(text) > 120:
                text = text[:117] + "..."
            return text

    return ""


def build_dataframe(
    episodes: list[dict],
    people: dict[str, dict],
    verbose: bool = False,
) -> pd.DataFrame:
    """Build the person-mention-per-episode DataFrame.

    Args:
        episodes: List of episode source dicts from ES.
        people: Dict of nconst -> person source dict from ES.
        verbose: If True, include extra columns.

    Returns:
        DataFrame with one row per person-mention per episode.
    """
    rows: list[dict] = []

    # Build a lookup: (episode_title) -> episode source for segment access
    episode_lookup: dict[str, dict] = {}
    for ep in episodes:
        episode_lookup[ep.get("episode_title", "")] = ep

    episode_titles = {ep.get("episode_title", "") for ep in episodes}

    for nconst, person in people.items():
        mentions = person.get("episode_mentions", [])
        for mention in mentions:
            ep_title = mention.get("episode_title", "")
            if ep_title not in episode_titles:
                continue

            ep = episode_lookup.get(ep_title, {})
            mention_count = mention.get("mention_count", 0)
            known_for = person.get("known_for_titles", [])
            known_for_str = ", ".join(known_for) if known_for else ""

            # Hosts are not stored in ES; leave blank for now — the website
            # can parse them from descriptions later.
            episode_hosts = ""

            # Get first mention timestamp from entity files if available
            first_ts = _get_first_timestamp(ep, person.get("name", ""))

            sample_quote = extract_sample_quote(
                ep, person.get("name", ""), first_ts
            )

            row: dict = {
                "episode_title": ep_title,
                "episode_date": mention.get("episode_date", ""),
                "episode_hosts": episode_hosts,
                "person_name": person.get("name", ""),
                "imdb_id": nconst,
                "primary_role": person.get("category", "other"),
                "mention_count": mention_count,
                "known_for_titles": known_for_str,
                "first_mention_timestamp": _format_timestamp(first_ts),
                "sample_quote": sample_quote,
            }

            if verbose:
                row["professions"] = ", ".join(person.get("professions", []))
                row["total_mentions_all_episodes"] = person.get("total_mentions", 0)

            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Sort by episode_date desc, then mention_count desc
    df = df.sort_values(
        ["episode_date", "mention_count"], ascending=[False, False]
    ).reset_index(drop=True)

    return df


def _get_first_timestamp(episode: dict, person_name: str) -> float | None:
    """Search local entity files for the first mention timestamp of a person.

    Falls back to scanning episode segments if entity files aren't available.

    Args:
        episode: Episode source dict.
        person_name: Person's display name.

    Returns:
        Timestamp in seconds, or None.
    """
    slug = episode.get("slug", "")
    if slug:
        entity_path = config.ENTITIES_DIR / f"{slug}.json"
        if entity_path.exists():
            try:
                with open(entity_path, "r", encoding="utf-8") as f:
                    entity_data = json.load(f)
                for p in entity_data.get("people", []):
                    if p.get("name", "").lower() == person_name.lower():
                        timestamps = p.get("timestamps", [])
                        if timestamps:
                            return timestamps[0].get("start")
            except (json.JSONDecodeError, KeyError):
                pass

    # Fallback: scan segments for name
    last_name = person_name.split()[-1].lower() if person_name else ""
    for seg in episode.get("segments", []):
        if last_name and last_name in seg.get("text", "").lower():
            return seg.get("start")

    return None


def _format_timestamp(seconds: float | None) -> str:
    """Format seconds as MM:SS or empty string.

    Args:
        seconds: Timestamp in seconds.

    Returns:
        Formatted string like "12:34" or "".
    """
    if seconds is None:
        return ""
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins}:{secs:02d}"


def print_summary(df: pd.DataFrame, num_episodes: int) -> None:
    """Print a summary header with aggregate statistics.

    Args:
        df: The results DataFrame.
        num_episodes: Number of episodes queried.
    """
    if df.empty:
        logger.info("No results found.")
        return

    total_people = df["person_name"].nunique()
    role_counts = df.drop_duplicates(subset=["person_name", "imdb_id"]).groupby("primary_role").size()

    directors = role_counts.get("director", 0)
    actors = role_counts.get("actor", 0)
    cinematographers = role_counts.get("cinematographer", 0)
    other = role_counts.get("other", 0)

    print("\n" + "=" * 70)
    print("PENDIX — Test Results Summary")
    print("=" * 70)
    print(f"  Episodes queried:    {num_episodes}")
    print(f"  Total people found:  {total_people}")
    print(f"  Breakdown:           {directors} directors, {actors} actors, "
          f"{cinematographers} cinematographers, {other} other")
    print("=" * 70 + "\n")


def output_results(
    df: pd.DataFrame, output_format: str, num_episodes: int
) -> None:
    """Display or save the DataFrame in the requested format.

    Args:
        df: Results DataFrame.
        output_format: One of "terminal", "csv", "json".
        num_episodes: Number of episodes queried (for summary header).
    """
    print_summary(df, num_episodes)

    if df.empty:
        return

    if output_format == "terminal":
        # Truncate wide columns for terminal display
        display_df = df.copy()
        for col in ("known_for_titles", "sample_quote"):
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: (x[:40] + "...") if isinstance(x, str) and len(x) > 43 else x
                )
        print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False))
        print()

    elif output_format == "csv":
        output_path = config.DATA_DIR / "test_output.csv"
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved CSV to %s", output_path)

    elif output_format == "json":
        output_path = config.DATA_DIR / "test_output.json"
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2, force_ascii=False)
        logger.info("Saved JSON to %s", output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Query Elasticsearch and display pipeline results as a DataFrame.",
    )
    parser.add_argument(
        "--episodes",
        default="latest",
        help=(
            "Which episodes to query. Options: 'latest' (default, 2 most recent), "
            "'all', or a comma-separated list of episode slugs or dates "
            "(e.g. '2026-02-10,2026-02-03')"
        ),
    )
    parser.add_argument(
        "--output",
        choices=["terminal", "csv", "json"],
        default="terminal",
        help="Output format (default: terminal)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show additional detail like professions and cross-episode totals",
    )
    parser.add_argument(
        "--es-url",
        default=config.ELASTICSEARCH_URL,
        help=f"Elasticsearch URL (default: {config.ELASTICSEARCH_URL})",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: query ES, build DataFrame, and output results."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    es = get_es_client(args.es_url)

    episodes = fetch_episodes(es, args.episodes)
    if not episodes:
        logger.warning("No episodes found. Nothing to display.")
        sys.exit(0)

    people = fetch_people(es)
    logger.info("Loaded %d people from index.", len(people))

    df = build_dataframe(episodes, people, verbose=args.verbose)
    output_results(df, args.output, num_episodes=len(episodes))


if __name__ == "__main__":
    main()
