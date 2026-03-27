# Scripts Guide

> Note: If you download the `cached` folder from Google Drive, it already contains:
> - Mapping shards in `cached/mapping/`
> - PMXT parquet files that have already been truncated in `cached/pmxt/`
> Transfer the downloaded `cached` folder to `data/cached` in the project root, and you should be able to start from the "Inspecting PMXT Files" section below.
> In that case, you can skip mapping build + download unless you need fresh data.

This folder contains utility scripts for downloading, inspecting, and filtering Polymarket data.

Run commands from the project root.

## Prerequisites

Install project dependencies first:

```bash
uv sync -U
```

If you run scripts directly with Python instead of uv, use a Python environment that includes:
- requests
- beautifulsoup4
- tqdm
- pyarrow

## Typical Workflow

1. Build daily market mapping shards from Gamma.
2. Download PMXT parquet files (each downloaded file is truncated automatically).
3. (Optional) Run manual truncation for backfills/reprocessing.
4. (Optional) Inspect parquet schema/sample rows.

## Recommended Run Order

```bash
# 1) Build mapping first (pick the date window you will download)
uv run python scripts/build_updown_market_mapping.py \
  --start-date 2026-02-21 \
  --end-date 2026-03-11

# 2) Download next (auto-truncates each downloaded parquet file)
uv run python scripts/download_polymarket.py --start-date 2026-02-21
```

Why this order:
- `download_polymarket.py` calls truncation immediately after each successful download.
- Truncation depends on mapping shards for the file's date.

## Script Reference

### build_updown_market_mapping.py

Builds slug-to-market metadata shards used for filtering PMXT files. Output shards are written to:

- data/cached/mapping/gamma_updown_markets_YYYY-MM-DD.json

Examples:

```bash
# Build mapping for inferred PMXT range -> today
uv run python scripts/build_updown_market_mapping.py

# Build mapping for explicit date range
uv run python scripts/build_updown_market_mapping.py \
  --start-date 2026-02-21 \
  --end-date 2026-03-11

# Restrict assets/resolutions
uv run python scripts/build_updown_market_mapping.py \
  --assets btc,eth,sol \
  --resolutions 5m,15m,1h

# Estimate work only (no API writes)
uv run python scripts/build_updown_market_mapping.py --dry-run
```

### download_polymarket.py

Downloads all Polymarket archive parquet files from https://archive.pmxt.dev/Polymarket into:

- data/cached/pmxt

After each successful download, it automatically runs `truncate_pmxt_by_tokens.py --file <downloaded_file>`.

Examples:

```bash
# Download everything found in archive pages
uv run python scripts/download_polymarket.py

# Only process files dated on/after 2026-03-01
uv run python scripts/download_polymarket.py --start-date 2026-03-01
```

Notes:
- Existing files are skipped.
- Retries are built in for page fetch and file download errors.

### truncate_pmxt_by_tokens.py

Filters PMXT parquet files in place using condition IDs from mapping data.

Default input:
- source: data/cached/pmxt
- mapping: data/cached/mapping

Examples:

```bash
# Truncate all PMXT files using daily mapping shards
uv run python scripts/truncate_pmxt_by_tokens.py

# Dry run (show retained rows, do not write)
uv run python scripts/truncate_pmxt_by_tokens.py --dry-run

# Process only files on/after a date
uv run python scripts/truncate_pmxt_by_tokens.py --start-date 2026-03-01

# Process one specific parquet file
uv run python scripts/truncate_pmxt_by_tokens.py \
  --file data/cached/pmxt/polymarket_orderbook_2026-03-10T14.parquet

# Keep empty parquet files instead of deleting them
uv run python scripts/truncate_pmxt_by_tokens.py --keep-empty
```

Important behavior:
- Runs in place (original files are overwritten or deleted if empty and `--keep-empty` is not set).
- If `--mapping` points to a directory, it loads per-day shards.
- If `--mapping` points to a single JSON file, it uses legacy single-file mode.

### inspect_pmxt_orderbook_structure.py

Inspects parquet metadata, schema, row samples, and decoded JSON payload structure.

Examples:

```bash
# Recommended: inspect cached PMXT files
uv run python scripts/inspect_pmxt_orderbook_structure.py \
  --data-dir data/cached/pmxt \
  --max-files 3 \
  --sample-rows 5

# Custom file pattern
uv run python scripts/inspect_pmxt_orderbook_structure.py \
  --data-dir data/cached/pmxt \
  --pattern "polymarket_orderbook_2026-03-*.parquet"
```

## Troubleshooting

- No files found:
  - Confirm you are running from project root.
  - Confirm paths under data/cached exist.
- Module import errors:
  - Run `uv sync -U` and rerun with `uv run python ...`.
- Mapping shard missing for a given date:
  - Rebuild mapping with `build_updown_market_mapping.py` for the relevant date range.