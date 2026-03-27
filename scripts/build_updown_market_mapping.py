"""Build static slug -> clobTokenIds mapping for crypto updown markets.

This script deduplicates slugs across time buckets and fetches market data from
Gamma API with minimal redundant requests. Outputs a static mapping file that can
be reused by truncation and analysis scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

# Add workspace to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.reference.gamma import (
    _extract_asset_ids,
    _fetch_markets_by_slug,
    _resolution_to_seconds,
)

DEFAULT_PMXT_DIR = Path("data") / "cached" / "pmxt"
DEFAULT_OUTPUT_DIR = Path("data") / "cached" / "mapping"
DEFAULT_ASSETS = "btc,eth,xrp,sol"
DEFAULT_RESOLUTIONS = "15m, 5m"
POLYMARKET_GAMMA_API_URL = "https://gamma-api.polymarket.com"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.5
SLUG_BATCH_SIZE = 200


@dataclass(frozen=True)
class ExpandedMarketEntry:
    """Extended cache entry with both clobTokenIds and conditionId."""

    clobTokenIds: list[str]
    conditionId: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build static slug -> clobTokenIds mapping for crypto updown markets.",
    )
    parser.add_argument(
        "--pmxt-dir",
        type=Path,
        default=DEFAULT_PMXT_DIR,
        help="Directory containing PMXT hourly parquet files (default: data/pmxt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for daily mapping shards (default: data/cached/mapping).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for slug enumeration (YYYY-MM-DD). Defaults to first date in PMXT directory.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for slug enumeration (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default=DEFAULT_ASSETS,
        help="Comma-separated crypto symbols to include (default: common set).",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default=DEFAULT_RESOLUTIONS,
        help="Comma-separated up-down resolutions (default: 5m,15m,1h,1d).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate API calls and tokens without fetching or writing.",
    )
    return parser.parse_args()


def parse_comma_list(raw_value: str) -> list[str]:
    """Parse a comma-separated argument into normalized values."""
    values = [part.strip().lower() for part in raw_value.split(",") if part.strip()]
    return sorted(set(values))


def parse_hour_from_filename(file_path: Path) -> datetime:
    """Extract UTC hour from polymarket_orderbook_YYYY-MM-DDTHH.parquet."""
    stem = file_path.stem
    prefix = "polymarket_orderbook_"
    if not stem.startswith(prefix):
        msg = f"Unsupported filename format: {file_path.name}"
        raise ValueError(msg)

    hour_raw = stem.removeprefix(prefix)
    return datetime.strptime(hour_raw, "%Y-%m-%dT%H").replace(tzinfo=UTC)


def find_pmxt_hour_range(pmxt_dir: Path) -> tuple[datetime, datetime] | None:
    """Scan PMXT directory and find min/max UTC hour boundaries."""
    pmxt_files = sorted(pmxt_dir.rglob("polymarket_orderbook_*.parquet"))
    if not pmxt_files:
        return None

    hours = []
    for pmxt_file in pmxt_files:
        try:
            hour = parse_hour_from_filename(pmxt_file)
            hours.append(hour)
        except ValueError:
            continue

    if not hours:
        return None

    return min(hours), max(hours)


def load_legacy_cache(shard_file: Path) -> dict[str, Any]:
    """Load existing shard cache, supporting both old and new schemas."""
    if not shard_file.exists():
        return {}

    try:
        payload = json.loads(shard_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    return payload


def save_cache(shard_file: Path, cache_data: dict[str, Any]) -> None:
    """Persist mapping shard to disk with expanded schema."""
    shard_file.parent.mkdir(parents=True, exist_ok=True)
    # Convert dataclass entries to dicts for JSON serialization
    serializable_data = {
        key: value if isinstance(value, dict) else value.__dict__
        for key, value in cache_data.items()
    }
    shard_file.write_text(
        json.dumps(serializable_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def shard_date_from_slug(slug: str) -> str | None:
    """Return UTC date shard key (YYYY-MM-DD) parsed from slug timestamp."""
    try:
        parts = slug.split("-")
        if len(parts) < 4:
            return None
        ts = int(parts[-1])
        return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")
    except (ValueError, IndexError):
        return None


def shard_file_for_date(output_dir: Path, date_key: str) -> Path:
    """Build shard filename for a date key."""
    return output_dir / f"gamma_updown_markets_{date_key}.json"


def chunk_list(values: list[str], chunk_size: int) -> list[list[str]]:
    """Split a list into fixed-size chunks."""
    if chunk_size <= 0:
        msg = "chunk_size must be positive"
        raise ValueError(msg)
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]



def _asset_to_slug_name(asset: str) -> str:
    """Convert asset symbol to full name for slugs."""
    return {
        "btc": "bitcoin",
        "eth": "ethereum",
        "sol": "solana",
    }.get(asset.lower(), asset)


def enumerate_unique_slugs(
    min_hour: datetime,
    max_hour: datetime,
    assets: list[str],
    resolutions: list[str],
) -> set[str]:
    """Generate all unique time-bucket slugs for asset/resolution combinations.

    Important: iterate each resolution on its own cadence. Stepping hourly and
    aligning (old behavior) only yields one 5m and one 15m slug per hour.
    """
    slugs: set[str] = set()

    min_ts = int(min_hour.timestamp())
    max_ts = int(max_hour.timestamp())

    for resolution in resolutions:
        seconds = _resolution_to_seconds(resolution)
        first_bucket_ts = (min_ts // seconds) * seconds

        bucket_ts = first_bucket_ts
        while bucket_ts <= max_ts:
            bucket_dt = datetime.fromtimestamp(bucket_ts, tz=UTC)

            for asset in assets:
                asset_name = _asset_to_slug_name(asset)
                if resolution == "1d":
                    slug = f"{asset_name}-up-or-down-on-{bucket_dt.strftime('%B-%d-%Y').lower()}"
                else:
                    slug = f"{asset}-updown-{resolution}-{bucket_ts}"

                slugs.add(slug)

            bucket_ts += seconds

    return slugs


def fetch_market_data_for_slug(
    slug: str,
    utctime: int,
    resolution: str,
) -> tuple[list[str], str | None]:
    """Fetch market data and extract clobTokenIds and conditionId.

    Returns
    -------
    tuple[list[str], str | None]
        clobTokenIds list and conditionId (or None if not found).
    """
    response_data = _fetch_markets_by_slug(
        market_slug=slug,
        utctime=utctime,
        resolution=resolution,
    )
    if not response_data:
        return [], None

    first_market = response_data[0]
    if not isinstance(first_market, dict):
        return [], None

    clobTokenIds = _extract_asset_ids(response_data)
    conditionId = first_market.get("conditionId")
    if not isinstance(conditionId, str) or not conditionId:
        conditionId = None

    return clobTokenIds, conditionId


def build_market_mapping(
    pmxt_dir: Path,
    assets: list[str],
    resolutions: list[str],
    output_dir: Path,
    min_hour: datetime,
    max_hour: datetime,
    dry_run: bool = False,
) -> tuple[int, int, int, int]:
    """Build mapping using constructed slugs (since API pagination doesn't return updown markets).

    Returns
    -------
    tuple[int, int, int, int]
        total_unique_slugs, cache_hits, api_calls, api_misses
    """
    print(f"Building mapping for date range: {min_hour.date()} to {max_hour.date()}")

    # Use constructed slugs since API pagination doesn't return updown markets
    unique_slugs = enumerate_unique_slugs(min_hour, max_hour, assets, resolutions)
    print(f"Enumerated {len(unique_slugs)} unique slugs")

    output_dir.mkdir(parents=True, exist_ok=True)

    cache_hits = 0
    api_calls = 0
    api_misses = 0
    slugs_to_fetch = []

    shard_caches: dict[str, dict[str, Any]] = {}

    slug_dates: dict[str, str] = {}

    for slug in sorted(unique_slugs):
        date_key = shard_date_from_slug(slug)
        if date_key is None:
            slugs_to_fetch.append(slug)
            continue

        slug_dates[slug] = date_key

        if date_key not in shard_caches:
            shard_path = shard_file_for_date(output_dir, date_key)
            shard_caches[date_key] = load_legacy_cache(shard_path)

        if slug in shard_caches[date_key]:
            cache_hits += 1
        else:
            slugs_to_fetch.append(slug)

    print(f"Cache hits: {cache_hits}")
    print(f"Slugs to fetch: {len(slugs_to_fetch)}")

    daily_fetch_counts: dict[str, int] = {}
    for slug in slugs_to_fetch:
        date_key = slug_dates.get(slug)
        if date_key is None:
            continue
        daily_fetch_counts[date_key] = daily_fetch_counts.get(date_key, 0) + 1

    if daily_fetch_counts:
        print("\nDaily fetch plan:")
        for date_key in sorted(daily_fetch_counts):
            print(f"  {date_key}: {daily_fetch_counts[date_key]} slugs")

    if dry_run:
        print(f"\nDRY RUN: Would fetch {len(slugs_to_fetch)} slugs from Gamma API")
        return len(unique_slugs), cache_hits, len(slugs_to_fetch), 0

    if not slugs_to_fetch:
        print("No missing slugs to fetch. Mapping shards are up to date.")
        return len(unique_slugs), cache_hits, 0, 0

    daily_queue: dict[str, list[str]] = {}
    for slug in sorted(slugs_to_fetch):
        date_key = slug_dates.get(slug)
        if date_key is None:
            continue
        daily_queue.setdefault(date_key, []).append(slug)

    # Fetch missing slugs directly via slug query (API pagination doesn't return updown markets)
    total_days = len(daily_queue)
    global_idx = 0
    total_to_fetch = len(slugs_to_fetch)

    for day_idx, date_key in enumerate(sorted(daily_queue.keys()), start=1):
        day_slugs = daily_queue[date_key]
        print(f"\n[{day_idx}/{total_days}] Processing day {date_key} ({len(day_slugs)} slugs)")

        day_batches = chunk_list(day_slugs, SLUG_BATCH_SIZE)

        with tqdm(total=len(day_slugs), desc=f"{date_key}", unit="slug", leave=False) as pbar:
            for batch in day_batches:
                try:
                    params: list[tuple[str, str | int]] = [("slug", slug) for slug in batch]
                    # Keep limit proportional to batch size to avoid truncation.
                    params.append(("limit", max(50, len(batch) * 5)))

                    response = requests.get(f"{POLYMARKET_GAMMA_API_URL}/markets", params=params, timeout=10)
                    response.raise_for_status()
                    markets = response.json()

                    markets_by_slug: dict[str, list[dict[str, Any]]] = {}
                    if isinstance(markets, list):
                        for market in markets:
                            if not isinstance(market, dict):
                                continue
                            market_slug = market.get("slug")
                            if isinstance(market_slug, str) and market_slug:
                                markets_by_slug.setdefault(market_slug, []).append(market)

                    if date_key not in shard_caches:
                        shard_caches[date_key] = load_legacy_cache(shard_file_for_date(output_dir, date_key))

                    for slug in batch:
                        global_idx += 1
                        matched_markets = markets_by_slug.get(slug, [])

                        if matched_markets:
                            first_market = matched_markets[0]
                            clobTokenIds = _extract_asset_ids([first_market])
                            conditionId = first_market.get("conditionId")
                            if not isinstance(conditionId, str) or not conditionId:
                                conditionId = None
                            shard_caches[date_key][slug] = {
                                "clobTokenIds": clobTokenIds,
                                "conditionId": conditionId,
                            }
                            pbar.set_postfix(status="HIT", global_progress=f"{global_idx}/{total_to_fetch}")
                        else:
                            shard_caches[date_key][slug] = {"clobTokenIds": [], "conditionId": None}
                            api_misses += 1
                            pbar.set_postfix(status="MISS", global_progress=f"{global_idx}/{total_to_fetch}")

                        pbar.update(1)

                    api_calls += 1
                    save_cache(shard_file_for_date(output_dir, date_key), shard_caches[date_key])
                except requests.RequestException as exc:
                    if date_key not in shard_caches:
                        shard_caches[date_key] = load_legacy_cache(shard_file_for_date(output_dir, date_key))

                    for slug in batch:
                        global_idx += 1
                        shard_caches[date_key][slug] = {"clobTokenIds": [], "conditionId": None}
                        api_misses += 1
                        pbar.update(1)

                    save_cache(shard_file_for_date(output_dir, date_key), shard_caches[date_key])
                    pbar.set_postfix(status="ERROR", global_progress=f"{global_idx}/{total_to_fetch}")
                    print(f"  ERROR batch ({len(batch)} slugs): {exc}")

                time.sleep(0.5)  # Rate limiting

    print(f"\nSaved {len(shard_caches)} daily shard files to {output_dir}")

    return len(unique_slugs), cache_hits, api_calls, api_misses


def validate_inputs(pmxt_dir: Path, assets: list[str], resolutions: list[str]) -> int:
    """Validate inputs and print user-facing errors."""
    if not pmxt_dir.exists() or not pmxt_dir.is_dir():
        print(f"Error: PMXT directory not found: {pmxt_dir}")
        return 1
    if not assets:
        print("Error: at least one asset is required")
        return 1
    if not resolutions:
        print("Error: at least one resolution is required")
        return 1
    return 0


def main() -> int:
    """Run market mapping builder."""
    args = parse_args()

    pmxt_dir = args.pmxt_dir.resolve()
    output_dir = args.output_dir.resolve()
    assets = parse_comma_list(args.assets)
    resolutions = parse_comma_list(args.resolutions)

    # Determine date range
    if args.start_date:
        try:
            min_hour = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            print(f"Error: invalid start date format '{args.start_date}', use YYYY-MM-DD")
            return 1
    else:
        # Default to first date in PMXT directory
        hour_range = find_pmxt_hour_range(pmxt_dir)
        if hour_range is None:
            print(f"Error: no PMXT parquet files found in {pmxt_dir}")
            return 1
        min_hour, _ = hour_range

    if args.end_date:
        try:
            max_hour = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=UTC)
            # Set to end of day
            max_hour = max_hour.replace(hour=23, minute=59, second=59)
        except ValueError:
            print(f"Error: invalid end date format '{args.end_date}', use YYYY-MM-DD")
            return 1
    else:
        # Default to today at end of day
        max_hour = datetime.now(UTC).replace(hour=23, minute=59, second=59)

    print(f"PMXT directory: {pmxt_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Date range: {min_hour.date()} to {max_hour.date()}")
    print(f"Assets: {', '.join(assets)}")
    print(f"Resolutions: {', '.join(resolutions)}")
    print(f"Dry run: {args.dry_run}\n")

    validation_code = validate_inputs(
        pmxt_dir=pmxt_dir,
        assets=assets,
        resolutions=resolutions,
    )
    if validation_code != 0:
        return validation_code

    total_slugs, cache_hits, api_calls, api_misses = build_market_mapping(
        pmxt_dir=pmxt_dir,
        assets=assets,
        resolutions=resolutions,
        output_dir=output_dir,
        min_hour=min_hour,
        max_hour=max_hour,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 70)
    print("Market Mapping Summary:")
    print(f"  Unique slugs:    {total_slugs}")
    print(f"  Cache hits:      {cache_hits}")
    if not args.dry_run:
        print(f"  API calls made:  {api_calls}")
        print(f"  API misses:      {api_misses}")
    else:
        print(f"  API calls (est): {api_calls}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nMapping builder interrupted by user.")
        sys.exit(1)
