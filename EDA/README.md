# Polymarket Crypto Up/Down EDA

This package builds a RAM-efficient exploratory research pipeline for Polymarket crypto up/down contracts. The workflow is designed for cross-coin and cross-resolution comparison without loading the full raw dataset into memory.

## Layout

```text
EDA/
├── core/
│   ├── config.py
│   ├── io.py
│   ├── schema.py
│   └── utils.py
├── pipeline/
│   ├── analysis.py
│   └── features.py
├── render/
│   └── plots.py
├── output/
│   ├── cache/
│   └── figures/
└── run_eda.py
```

Only `EDA/run_eda.py` is meant to be executed directly. The remaining implementation code lives in subpackages for clearer separation of concerns.

## Execution

Default execution mode is `full`.

```bash
uv run python EDA/run_eda.py
uv run python EDA/run_eda.py --mode sample
uv run python EDA/run_eda.py --mode full
```

Sample mode is still the first validation step. It processes a small bounded subset of shared hours, confirms schemas and joins, and runs the entire workflow end to end before a full-scale run.

## Pipeline Design

The pipeline is intentionally staged:

1. Inventory parquet files and schema variants.
2. Build a deduplicated `market_id x token_id` universe from `mapping`.
3. Process `book_snapshot` and `price_change` one hour at a time into compact cached summaries.
4. Run cross-coin and cross-resolution analytics from those cached summaries.
5. Generate presentation-quality figures and a synthesized research note from those analysis results.

This keeps peak memory usage bounded while preserving a clean full-run path.

## Output Conventions

- Cached tables live under `EDA/output/cache/<mode>/`.
- Figures live under `EDA/output/figures/<mode>/`.
- The synthesized research note is written to `EDA/insights.md`.
Key cached artifacts include:

- `file_inventory.parquet`
- `schema_summary.parquet`
- `market_dimension.parquet`
- `book_hourly/*.parquet`
- `price_hourly/*.parquet`
- `tables/*.parquet`

## Visualization Rules

- Comparison heatmaps are rendered against the complete coin x resolution universe.
- The expected universe is 7 coins by 3 resolutions: `5m`, `15m`, `4h`.
- Sparse or inactive combinations remain visible as `NA` or `No data`; they are not silently dropped from charts.
- Resolution ordering is fixed to `5m`, `15m`, `4h` to keep cross-figure comparisons consistent.

## Notes

- `mapping` is scanned with union-safe settings because the schema drifts across hourly files.
- `price_change` may contain chunked files for a single hour, so batch discovery groups files by logical hour.
- Full mode is the default output path, but sample mode should still be used first when validating a new environment or code change.
