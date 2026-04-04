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
- If `--mapping` points to a single JSON file, it loads that file directly.

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

### prepare_market_backtest_dataset.py

Builds backtest-ready feature and resolution artifacts from PMXT shards.

Key behavior:
- Uses slug timestamp windows from mapping shards to schedule only likely parquet files.
- Supports deterministic first-N mapping market selection with `--max-markets`.
- Creates isolated outputs per prefix run under:
  - `data/cached/pmxt_backtest/runs/<market-slug-prefix>/features/...`
  - `data/cached/pmxt_backtest/runs/<market-slug-prefix>/resolution/resolution_frame.parquet`
  - `data/cached/pmxt_backtest/runs/<market-slug-prefix>/manifest.json`

Examples:

```bash
# Build one isolated run for BTC 15m markets
uv run python scripts/prepare_market_backtest_dataset.py \
  --market-slug-prefix btc-updown-15m

# Build first 10 markets for fast iteration
uv run python scripts/prepare_market_backtest_dataset.py \
  --market-slug-prefix btc-updown-15m \
  --max-markets 10

# Build a different prefix into a separate run root
uv run python scripts/prepare_market_backtest_dataset.py \
  --market-slug-prefix eth-updown-15m \
  --max-markets 10
```

Notes:
- Run one prefix per command to keep feature sets independent.
- Pass each run's manifest to the backtester when evaluating that prefix.
- If you omit `--market-slug-prefix`, output is written to the base output directory.

### build_resolution_from_mapping_vectors.py

Builds a backtester-compatible `resolution_frame.parquet` directly from Gamma mapping
binary vectors:

- `[1,0]` means first CLOB token wins
- `[0,1]` means second CLOB token wins

Output schema matches the backtester expectations:
- `market_id`
- `resolved_at`
- `winning_asset_id`
- `winning_outcome`
- `fees_enabled_market`
- `settlement_source`
- `settlement_confidence`
- `settlement_evidence_ts`

Examples:

```bash
# Rebuild resolution parquet for an existing prepared run
uv run python scripts/build_resolution_from_mapping_vectors.py \
  --run-dir data/cached/pmxt_backtest/runs/btc-updown-5m \
  --overwrite

# Dry run with explicit mapping and output paths
uv run python scripts/build_resolution_from_mapping_vectors.py \
  --mapping-dir data/cached/mapping \
  --output-path data/cached/pmxt_backtest/runs/btc-updown-5m/resolution/resolution_frame.parquet \
  --features-root data/cached/pmxt_backtest/runs/btc-updown-5m/features \
  --dry-run
```

Notes:
- If `--features-root` is provided (or inferred via `--run-dir`), only markets present in
  prepared features are included.
- If both `--market-ids-file` and `--features-root` are provided, the script uses the
  intersection of both filters.
- By default, the script refuses to overwrite an existing output unless `--overwrite` is set.

## Troubleshooting

- No files found:
  - Confirm you are running from project root.
  - Confirm paths under data/cached exist.
- Module import errors:
  - Run `uv sync -U` and rerun with `uv run python ...`.
- Mapping shard missing for a given date:
  - Rebuild mapping with `build_updown_market_mapping.py` for the relevant date range.