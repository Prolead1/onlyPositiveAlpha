"""Truncate PMXT parquet files to crypto updown markets using static token mapping.

This script loads the market mapping built by build_updown_market_mapping.py,
flattens the clobTokenIds, and filters PMXT parquet files to retain only those tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Add workspace to path for imports (not currently needed, but for consistency)
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass(frozen=True)
class FilterOptions:
    """Options for parquet filtering and output writing."""

    codec: str
    compression_level: int
    dry_run: bool
    keep_empty: bool


DEFAULT_SOURCE_DIR = Path("data") / "cached" / "pmxt"
DEFAULT_MAPPING_DIR = Path("data") / "cached" / "mapping"
LEGACY_MAPPING_FILE = Path("data") / "cached" / "gamma_updown_markets.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter PMXT parquet files to retained token IDs only.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Source directory with PMXT hourly parquet files (default: data/pmxt).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the original files before modifying them.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_DIR,
        help="Path to mapping directory (daily shards) or legacy single JSON file (default: data/cached/mapping).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date in ISO format (YYYY-MM-DD). Only process files on or after this date.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to a specific parquet file to truncate (overrides --start-date if provided).",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="zstd",
        help="Output parquet codec (default: zstd).",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=9,
        help="Output parquet compression level (default: 9).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate retained rows without writing output files.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Write empty output files when no target tokens are found in that hour.",
    )
    return parser.parse_args()


def load_mapping(mapping_file: Path) -> dict[str, dict[str, object]]:
    """Load nested market mapping from a single JSON file."""
    if not mapping_file.exists():
        raise FileNotFoundError(mapping_file)

    try:
        payload = json.loads(mapping_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: failed to parse mapping JSON: {exc}")
        raise

    if not isinstance(payload, dict):
        print(f"Error: expected dict at top level of mapping file, got {type(payload)}")
        raise TypeError(payload)

    return payload


def day_shard_from_timestamp(mapping_dir: Path, hour_ts: int) -> Path:
    """Resolve daily shard file path from UTC hour timestamp."""
    date_key = datetime.fromtimestamp(hour_ts, tz=UTC).strftime("%Y-%m-%d")
    return mapping_dir / f"gamma_updown_markets_{date_key}.json"


def load_mapping_for_hour(mapping_path: Path, hour_ts: int) -> dict[str, dict[str, object]]:
    """Load mapping for a specific hour from daily shard dir or legacy JSON path."""
    if mapping_path.is_file():
        return load_mapping(mapping_path)

    if mapping_path.is_dir():
        shard_file = day_shard_from_timestamp(mapping_path, hour_ts)
        if shard_file.exists():
            return load_mapping(shard_file)

        legacy_fallback = LEGACY_MAPPING_FILE.resolve()
        if legacy_fallback.exists():
            return load_mapping(legacy_fallback)

        raise FileNotFoundError(shard_file)

    raise FileNotFoundError(mapping_path)


def flatten_token_ids(mapping: dict[str, dict[str, object]]) -> set[str]:
    """Extract all unique conditionIds from mapping."""
    condition_ids: set[str] = set()

    for slug, entry in mapping.items():
        if not isinstance(entry, dict):
            continue

        condition_id = entry.get("conditionId")
        if isinstance(condition_id, str) and condition_id:
            condition_ids.add(condition_id)

    return condition_ids


def filter_mapping_by_hour(
    mapping: dict[str, dict[str, object]],
    hour_start_timestamp: int,
) -> set[str]:
    """Extract conditionIds only from markets that start in the given hour.
    
    Parameters
    ----------
    mapping : dict
        Market mapping with keys like "btc-updown-15m-<timestamp>"
    hour_start_timestamp : int
        Unix timestamp of hour start (e.g., 1771707600 for 2026-02-21 21:00:00)
    
    Returns
    -------
    set[str]
        ConditionIds for markets starting in [hour_start, hour_start+3600)
    """
    hour_end_timestamp = hour_start_timestamp + 3600
    condition_ids: set[str] = set()

    for slug, entry in mapping.items():
        if not isinstance(entry, dict):
            continue

        # Extract timestamp from slug (format: "crypto-updown-interval-<timestamp>")
        try:
            parts = slug.split("-")
            if len(parts) >= 4:
                ts = int(parts[-1])
                # Check if this market starts in the target hour
                if hour_start_timestamp <= ts < hour_end_timestamp:
                    condition_id = entry.get("conditionId")
                    if isinstance(condition_id, str) and condition_id:
                        condition_ids.add(condition_id)
        except (ValueError, IndexError):
            # Skip slugs that don't have valid timestamp
            pass

    return condition_ids


def collect_source_files(source_dir: Path) -> list[Path]:
    """Collect all PMXT parquet files."""
    source_files = sorted(source_dir.rglob("polymarket_orderbook_*.parquet"))
    return source_files


def filter_single_file(
    source_file: Path,
    allowed_market_ids: set[str],
    options: FilterOptions,
) -> tuple[int, int, int, int, int]:
    """Filter one parquet file and optionally write it.

    Returns
    -------
    tuple[int, int, int, int, int]
        total_rows, kept_rows, input_bytes, output_bytes, matched_markets
    """
    output_file = source_file  # Operate in-place

    parquet_meta = pq.ParquetFile(source_file).metadata
    total_rows = parquet_meta.num_rows if parquet_meta is not None else 0
    input_bytes = source_file.stat().st_size

    if not allowed_market_ids:
        if not options.dry_run and options.keep_empty:
            empty_table = pq.read_table(source_file).slice(0, 0)
            pq.write_table(
                empty_table,
                output_file,
                compression=options.codec,
                compression_level=options.compression_level,
            )
            return total_rows, 0, input_bytes, output_file.stat().st_size, 0
        return total_rows, 0, input_bytes, 0, 0

    dataset = ds.dataset(source_file, format="parquet")
    # Filter by market_id (which holds the conditionId from Gamma API)
    market_filter = ds.field("market_id").isin(sorted(allowed_market_ids))
    kept_rows = dataset.count_rows(filter=market_filter)

    if options.dry_run:
        filtered_table = dataset.to_table(filter=market_filter)
        matched_markets = len(set(filtered_table["market_id"].to_pylist())) if kept_rows > 0 else 0
        return total_rows, kept_rows, input_bytes, 0, matched_markets

    if kept_rows == 0 and not options.keep_empty:
        # If we don't keep empty files, we should delete the original
        source_file.unlink()
        return total_rows, kept_rows, input_bytes, 0, 0

    filtered_table = dataset.to_table(filter=market_filter)
    matched_markets = len(set(filtered_table["market_id"].to_pylist()))
    pq.write_table(
        filtered_table,
        output_file,
        compression=options.codec,
        compression_level=options.compression_level,
    )
    return total_rows, kept_rows, input_bytes, output_file.stat().st_size, matched_markets


def validate_inputs(source_dir: Path, market_ids: set[str]) -> int:
    """Validate inputs and print user-facing errors."""
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Error: source directory not found: {source_dir}")
        return 1
    if not market_ids:
        print("Error: no market IDs (conditionIds) found in mapping")
        return 1
    return 0


def parse_start_date(date_str: str | None) -> int | None:
    """Parse ISO format date string to UTC timestamp of start of that day.
    
    Parameters
    ----------
    date_str : str | None
        Date in ISO format (YYYY-MM-DD) or None
    
    Returns
    -------
    int | None
        Unix timestamp of 00:00:00 UTC on that date, or None if date_str is None
    """
    if date_str is None:
        return None

    try:
        year, month, day = map(int, date_str.split("-"))
        dt = datetime(year, month, day, 0, 0, 0, tzinfo=UTC)
        return int(dt.timestamp())
    except (ValueError, AttributeError):
        print(f"Error: invalid start date format '{date_str}'. Use YYYY-MM-DD")
        raise


def extract_hour_from_filename(filename: str) -> int | None:
    """Extract hour timestamp from filename like 'polymarket_orderbook_2026-02-21T21.parquet'.
    
    Returns
    -------
    int | None
        Unix timestamp of hour start (UTC), or None if parsing fails
    """
    try:
        # Format: polymarket_orderbook_YYYY-MM-DDTHH.parquet
        parts = filename.replace("polymarket_orderbook_", "").replace(".parquet", "")
        # parts = "2026-02-21T21"

        # Parse manually to ensure UTC interpretation
        # Split by T: ["2026-02-21", "21"]
        date_hour = parts.split("T")
        if len(date_hour) != 2:
            return None

        date_part = date_hour[0]  # "2026-02-21"
        hour_part = date_hour[1]  # "21"

        # Parse date
        year, month, day = map(int, date_part.split("-"))
        hour = int(hour_part)

        # Create UTC datetime
        dt = datetime(year, month, day, hour, 0, 0, tzinfo=UTC)
        return int(dt.timestamp())
    except (ValueError, AttributeError, IndexError):
        return None


def process_files(
    source_files: list[Path],
    source_dir: Path,
    mapping_path: Path,
    options: FilterOptions,
) -> tuple[int, int, int, int, int, int, int]:
    """Filter all source files and return aggregate metrics.

    Returns
    -------
    tuple[int, int, int, int, int, int, int]
        processed_files, written_files, skipped_files, total_rows, kept_rows, input_bytes, output_bytes
    """
    processed_files = 0
    modified_files = 0
    deleted_files = 0
    total_rows = 0
    kept_rows = 0
    input_bytes = 0
    output_bytes = 0

    print("\nFiltering PMXT files by market IDs (conditionIds)...")
    for index, source_file in enumerate(source_files, start=1):
        # Extract hour from filename and filter mapping for that hour
        hour_ts = extract_hour_from_filename(source_file.name)

        if hour_ts is not None:
            try:
                hour_mapping = load_mapping_for_hour(mapping_path, hour_ts)
            except (FileNotFoundError, json.JSONDecodeError, TypeError):
                hour_mapping = {}
            hour_market_ids = filter_mapping_by_hour(hour_mapping, hour_ts)
        else:
            hour_market_ids = set()

        print(
            f"[{index}/{len(source_files)}] {source_file.name} | "
            f"filtering by {len(hour_market_ids)} markets (hour-specific)"
        )

        file_total_rows, file_kept_rows, file_input_bytes, file_output_bytes, file_matched_markets = filter_single_file(
            source_file=source_file,
            allowed_market_ids=hour_market_ids,
            options=options,
        )

        file_dropped_rows = file_total_rows - file_kept_rows
        print(
            "  Rows: "
            f"total={file_total_rows}, kept={file_kept_rows}, dropped={file_dropped_rows} | "
            f"Markets matched: {file_matched_markets}"
        )

        processed_files += 1
        total_rows += file_total_rows
        kept_rows += file_kept_rows
        input_bytes += file_input_bytes
        output_bytes += file_output_bytes

        if options.dry_run:
            continue

        if file_output_bytes > 0:
            modified_files += 1
            print(f"  Modified: {source_file}")
        else:
            deleted_files += 1
            print(f"  Deleted: {source_file}")

    return (
        processed_files,
        modified_files,
        deleted_files,
        total_rows,
        kept_rows,
        input_bytes,
        output_bytes,
    )


def main() -> int:
    """Run PMXT truncation using static token mapping."""
    args = parse_args()

    source_dir = args.source.resolve()
    mapping_path = args.mapping.resolve()

    print(f"Source directory: {source_dir}")
    print(f"Mapping path: {mapping_path}")
    if args.start_date:
        print(f"Start date: {args.start_date}")
    print(f"Dry run: {args.dry_run}")
    print(f"Keep empty files: {args.keep_empty}\n")

    if not mapping_path.exists():
        print(f"Error: mapping path not found: {mapping_path}")
        return 1

    # Parse start date if provided
    start_date_ts = None
    if args.start_date:
        try:
            start_date_ts = parse_start_date(args.start_date)
        except ValueError:
            return 1

    if mapping_path.is_file():
        try:
            mapping = load_mapping(mapping_path)
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as exc:
            print(f"Error loading mapping: {exc}")
            return 1
        market_ids = flatten_token_ids(mapping)
        print(f"Loaded {len(mapping)} market entries from single file")
        print(f"Extracted {len(market_ids)} unique market IDs (conditionIds) from mapping\n")
    else:
        market_ids = {"daily-sharded-mapping"}
        print("Using daily sharded mapping directory\n")

    validation_code = validate_inputs(
        source_dir=source_dir,
        market_ids=market_ids,
    )
    if validation_code != 0:
        return validation_code

    source_files = collect_source_files(source_dir)
    if not source_files:
        print("No PMXT parquet files found in source directory.")
        return 0

    # If specific file is provided, filter to only that file
    if args.file is not None:
        file_path = args.file.resolve()
        if not file_path.exists():
            print(f"Error: specified file not found: {file_path}")
            return 1
        source_files = [file_path]
        print(f"Processing specific file: {file_path.name}\n")
    # Filter files by start date if provided
    elif start_date_ts is not None:
        filtered_files = []
        for f in source_files:
            file_hour_ts = extract_hour_from_filename(f.name)
            if file_hour_ts is not None and file_hour_ts >= start_date_ts:
                filtered_files.append(f)

        skipped_count = len(source_files) - len(filtered_files)
        if skipped_count > 0:
            print(f"Skipped {skipped_count} file(s) before start date {args.start_date}\n")
        source_files = filtered_files

    if not source_files:
        print("No PMXT parquet files found after applying date filter.")
        return 0

    options = FilterOptions(
        codec=args.codec,
        compression_level=args.compression_level,
        dry_run=args.dry_run,
        keep_empty=args.keep_empty,
    )

    (
        processed_files,
        modified_files,
        deleted_files,
        total_rows,
        kept_rows,
        input_bytes,
        output_bytes,
    ) = process_files(
        source_files=source_files,
        source_dir=source_dir,
        mapping_path=mapping_path,
        options=options,
    )

    print("\n" + "=" * 70)
    print("Truncation Summary:")
    print(f"  Files processed: {processed_files}")
    if not args.dry_run:
        print(f"  Files modified:   {modified_files}")
        print(f"  Files deleted:   {deleted_files}")
    print(f"  Rows total:      {total_rows}")
    print(f"  Rows kept:       {kept_rows}")
    print(f"  Rows dropped:    {total_rows - kept_rows}")
    print(f"  Input bytes:     {input_bytes}")
    if not args.dry_run:
        print(f"  Output bytes:    {output_bytes}")
        print(f"  Bytes saved:     {input_bytes - output_bytes}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nTruncation interrupted by user.")
        sys.exit(1)
