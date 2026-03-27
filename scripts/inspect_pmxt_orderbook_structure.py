"""Inspect the structure of downloaded Polymarket orderbook parquet files.

This script helps with first-pass data understanding by showing:
1) The discovered orderbook parquet files
2) File-level parquet metadata and schema
3) A small sample of rows and Python-level value shapes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "pmxt"
DEFAULT_PATTERN = "polymarket_orderbook_*.parquet"
MAX_KEY_PREVIEW = 5
MAX_PAYLOAD_SAMPLES = 2

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _format_arrow_type(data_type: pa.DataType, indent: int = 0) -> list[str]:
    """Return a readable recursive description of an Arrow data type."""
    pad = " " * indent
    lines: list[str] = []

    if pa.types.is_struct(data_type):
        lines.append(f"{pad}struct")
        for field in data_type:
            nullable = "nullable" if field.nullable else "required"
            lines.append(f"{pad}  - {field.name} ({nullable}):")
            lines.extend(_format_arrow_type(field.type, indent + 6))
        return lines

    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        lines.append(f"{pad}list")
        value_field = data_type.value_field
        lines.append(f"{pad}  value:")
        lines.extend(_format_arrow_type(value_field.type, indent + 4))
        return lines

    if pa.types.is_fixed_size_list(data_type):
        lines.append(f"{pad}fixed_size_list[{data_type.list_size}]")
        lines.append(f"{pad}  value:")
        lines.extend(_format_arrow_type(data_type.value_type, indent + 4))
        return lines

    lines.append(f"{pad}{data_type}")
    return lines


def _summarize_python_value(value: object) -> str:
    """Return a concise shape/type summary for sampled Python values."""
    if value is None:
        return "None"

    if isinstance(value, list):
        if not value:
            return "list(len=0)"
        first = value[0]
        return f"list(len={len(value)}, first_type={type(first).__name__})"

    if isinstance(value, dict):
        keys = list(value.keys())
        key_preview = ", ".join(str(k) for k in keys[:MAX_KEY_PREVIEW])
        if len(keys) > MAX_KEY_PREVIEW:
            key_preview += ", ..."
        return f"dict(keys=[{key_preview}])"

    return type(value).__name__


def _infer_nested_structure(
    value: JsonValue,
    indent: int = 0,
    max_list_items: int = 2,
) -> list[str]:
    """Infer and format nested structure from a sampled JSON-compatible value."""
    pad = " " * indent

    if isinstance(value, dict):
        lines = [f"{pad}object"]
        for key, nested_value in value.items():
            lines.append(f"{pad}  - {key}:")
            lines.extend(_infer_nested_structure(nested_value, indent + 6, max_list_items))
        return lines

    if isinstance(value, list):
        lines = [f"{pad}array(len={len(value)})"]
        if value:
            # Show structure from first few elements in case there are mixed shapes.
            limit = min(len(value), max_list_items)
            for idx in range(limit):
                lines.append(f"{pad}  element[{idx}]:")
                lines.extend(_infer_nested_structure(value[idx], indent + 6, max_list_items))
            if len(value) > limit:
                lines.append(f"{pad}  ...")
        return lines

    return [f"{pad}{type(value).__name__}"]


def _print_data_payload_structure(sample_df: pd.DataFrame) -> None:
    """Parse and print nested JSON payload structure from the data column, if present."""
    if "data" not in sample_df.columns:
        return

    parsed_payloads: list[dict[str, JsonValue]] = []
    for raw in sample_df["data"].dropna().tolist():
        if not isinstance(raw, str):
            continue
        try:
            parsed_raw = json.loads(raw)
        except json.JSONDecodeError:
            continue
        parsed = cast("object", parsed_raw)
        if isinstance(parsed, dict):
            cast_payload = cast("dict[str, JsonValue]", parsed)
            parsed_payloads.append(cast_payload)

    if not parsed_payloads:
        print("\nNo JSON payloads could be decoded from sampled data column.")
        return

    print("\nDecoded JSON payload structure from sampled data column:")
    for payload_idx, payload in enumerate(parsed_payloads[:MAX_PAYLOAD_SAMPLES], start=1):
        print(f"- payload[{payload_idx}]:")
        for line in _infer_nested_structure(payload, indent=4):
            print(line)


def _validate_args(data_dir: Path, max_files: int, sample_rows: int) -> None:
    """Validate CLI arguments and raise clear errors for invalid values."""
    if max_files <= 0:
        raise ValueError("--max-files must be > 0")
    if sample_rows <= 0:
        raise ValueError("--sample-rows must be > 0")
    if not data_dir.exists():
        msg = f"Data directory does not exist: {data_dir}"
        raise FileNotFoundError(msg)


def _inspect_selected_files(files: list[Path], sample_rows: int) -> list[str]:
    """Inspect selected files and return names whose schema mismatches the first file."""
    first_schema: pa.Schema | None = None
    mismatched_schema_files: list[str] = []

    for fp in files:
        schema = _inspect_single_file(fp, sample_rows=sample_rows)
        if first_schema is None:
            first_schema = schema
            continue
        if not schema.equals(first_schema, check_metadata=False):
            mismatched_schema_files.append(fp.name)

    return mismatched_schema_files


def _inspect_single_file(file_path: Path, sample_rows: int) -> pa.Schema:
    """Print metadata and schema details for a single parquet file."""
    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata
    schema = parquet_file.schema_arrow

    print(f"\n{'=' * 100}")
    print(f"File: {file_path.name}")
    print(f"Rows: {metadata.num_rows:,} | Row groups: {metadata.num_row_groups}")
    created_by = metadata.created_by or "Unknown"
    print(f"Created by: {created_by}")

    print("\nArrow schema (logical types):")
    for field in schema:
        nullable = "nullable" if field.nullable else "required"
        print(f"- {field.name} ({nullable}):")
        for line in _format_arrow_type(field.type, indent=2):
            print(line)

    print("\nParquet schema (physical storage):")
    print(parquet_file.schema)

    print(f"\nSample rows (up to {sample_rows}):")
    batches = parquet_file.iter_batches(batch_size=sample_rows)
    try:
        first_batch = next(batches)
    except StopIteration:
        print("(file has no rows)")
        return schema

    sample_table = pa.Table.from_batches([first_batch], schema=schema)
    sample_df = sample_table.to_pandas()

    print(sample_df.head(sample_rows).to_string(index=False, max_colwidth=80))

    print("\nSampled Python-level value shapes by column:")
    for column_name in sample_df.columns:
        non_null = sample_df[column_name].dropna()
        if non_null.empty:
            print(f"- {column_name}: all null in sample")
            continue
        summary = _summarize_python_value(non_null.iloc[0])
        print(f"- {column_name}: {summary}")

    _print_data_payload_structure(sample_df)

    return schema


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect structure of Polymarket orderbook parquet files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing downloaded parquet files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN,
        help="Glob pattern for selecting files inside --data-dir.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=3,
        help="Maximum number of files to inspect (sorted alphabetically).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="Number of rows to sample per file.",
    )
    return parser


def main() -> None:
    """Inspect schema and sample values from orderbook parquet files."""
    args = _build_parser().parse_args()

    data_dir: Path = args.data_dir
    pattern: str = args.pattern
    max_files: int = args.max_files
    sample_rows: int = args.sample_rows

    _validate_args(data_dir=data_dir, max_files=max_files, sample_rows=sample_rows)

    files = sorted(data_dir.glob(pattern))
    print(f"Searching in: {data_dir}")
    print(f"Pattern: {pattern}")
    print(f"Matched files: {len(files)}")

    if not files:
        print("No files matched. Adjust --data-dir or --pattern.")
        return

    selected_files = files[:max_files]
    print(f"Inspecting first {len(selected_files)} file(s):")
    for fp in selected_files:
        print(f"- {fp.name}")

    print(f"\n{'#' * 100}")
    print("STRUCTURE INSPECTION")
    print(f"{'#' * 100}")

    mismatched_schema_files = _inspect_selected_files(selected_files, sample_rows=sample_rows)

    print(f"\n{'#' * 100}")
    print("SCHEMA CONSISTENCY")
    print(f"{'#' * 100}")
    if not mismatched_schema_files:
        print("All inspected files share the same schema.")
    else:
        print("Schema differs from the first inspected file for:")
        for name in mismatched_schema_files:
            print(f"- {name}")


if __name__ == "__main__":
    main()
