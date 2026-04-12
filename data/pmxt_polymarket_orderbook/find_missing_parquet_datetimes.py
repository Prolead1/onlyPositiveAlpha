#!/usr/bin/env python3
"""Find missing parquet datetimes between the earliest and latest filenames."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


# Ordered from most specific to least specific so date-only matches do not win
# when the filename actually contains a full timestamp.
DEFAULT_DATETIME_PATTERNS: list[tuple[str, str]] = [
    (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%S"),
    (r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", "%Y-%m-%d_%H-%M-%S"),
    (r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}", "%Y-%m-%d_%H:%M:%S"),
    (r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", "%Y-%m-%dT%H:%M"),
    (r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}", "%Y-%m-%d_%H-%M"),
    (r"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}", "%Y-%m-%d_%H:%M"),
    (r"\d{4}-\d{2}-\d{2}T\d{2}", "%Y-%m-%dT%H"),
    (r"\d{4}-\d{2}-\d{2}_\d{2}", "%Y-%m-%d_%H"),
    (r"\d{8}T\d{6}", "%Y%m%dT%H%M%S"),
    (r"\d{8}_\d{6}", "%Y%m%d_%H%M%S"),
    (r"\d{14}", "%Y%m%d%H%M%S"),
    (r"\d{8}T\d{4}", "%Y%m%dT%H%M"),
    (r"\d{8}_\d{4}", "%Y%m%d_%H%M"),
    (r"\d{12}", "%Y%m%d%H%M"),
    (r"\d{10}", "%Y%m%d%H"),
    (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
    (r"\d{8}", "%Y%m%d"),
]

FREQ_PATTERN = re.compile(
    r"^\s*(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|"
    r"h|hr|hrs|hour|hours|d|day|days)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedFile:
    path: Path
    token: str
    timestamp: datetime
    prefix: str
    suffix: str
    extension: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find missing parquet datetimes between the earliest and latest filenames in a folder."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="raw",
        help="Directory containing parquet files. Defaults to ./raw",
    )
    parser.add_argument(
        "--freq",
        help=(
            "Expected cadence such as 1h, 15m, or 1d. If omitted, the script "
            "infers the smallest observed gap."
        ),
    )
    parser.add_argument(
        "--datetime-regex",
        help=(
            "Regex that extracts the datetime token from each filename stem. "
            "Use together with --datetime-format."
        ),
    )
    parser.add_argument(
        "--datetime-format",
        help=(
            "strptime/strftime format for the datetime token, such as "
            "%%Y%%m%%d_%%H%%M%%S. Use together with --datetime-regex."
        ),
    )
    return parser.parse_args()


def parse_frequency(freq_text: str) -> timedelta:
    match = FREQ_PATTERN.fullmatch(freq_text)
    if not match:
        raise ValueError(
            f"Unsupported --freq value: {freq_text!r}. Use forms like 1h, 15m, or 1d."
        )

    amount = int(match.group(1))
    if amount <= 0:
        raise ValueError("--freq must be greater than zero.")
    unit = match.group(2).lower()
    if unit.startswith("s"):
        return timedelta(seconds=amount)
    if unit.startswith("m"):
        return timedelta(minutes=amount)
    if unit.startswith("h"):
        return timedelta(hours=amount)
    return timedelta(days=amount)


def load_parquet_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    files = sorted(path for path in directory.glob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {directory}")
    return files


def try_parse_files(
    files: list[Path], token_regex: str, datetime_format: str
) -> list[ParsedFile] | None:
    parsed: list[ParsedFile] = []

    for path in files:
        matches = list(re.finditer(token_regex, path.stem))
        if len(matches) != 1:
            return None

        match = matches[0]
        token = match.group(0)
        try:
            timestamp = datetime.strptime(token, datetime_format)
        except ValueError:
            return None

        parsed.append(
            ParsedFile(
                path=path,
                token=token,
                timestamp=timestamp,
                prefix=path.stem[: match.start()],
                suffix=path.stem[match.end() :],
                extension=path.suffix,
            )
        )

    return parsed


def detect_datetime_parser(
    files: list[Path], user_regex: str | None, user_format: str | None
) -> tuple[str, str, list[ParsedFile]]:
    if bool(user_regex) ^ bool(user_format):
        raise ValueError("--datetime-regex and --datetime-format must be provided together.")

    if user_regex and user_format:
        parsed = try_parse_files(files, user_regex, user_format)
        if parsed is None:
            raise ValueError(
                "The supplied --datetime-regex/--datetime-format pair did not "
                "produce exactly one valid datetime per filename."
            )
        return user_regex, user_format, parsed

    for token_regex, datetime_format in DEFAULT_DATETIME_PATTERNS:
        parsed = try_parse_files(files, token_regex, datetime_format)
        if parsed is not None:
            return token_regex, datetime_format, parsed

    raise ValueError(
        "Could not detect a supported datetime pattern in the parquet filenames. "
        "Try passing --datetime-regex and --datetime-format explicitly."
    )


def infer_frequency(timestamps: list[datetime]) -> timedelta:
    unique_timestamps = sorted(set(timestamps))
    if len(unique_timestamps) < 2:
        raise ValueError("Need at least two distinct timestamps to infer the cadence.")

    deltas = [
        int((current - previous).total_seconds())
        for previous, current in zip(unique_timestamps, unique_timestamps[1:])
    ]
    positive_deltas = [delta for delta in deltas if delta > 0]
    if not positive_deltas:
        raise ValueError("Could not infer cadence because all timestamps are identical.")

    smallest_gap = min(positive_deltas)
    if all(delta % smallest_gap == 0 for delta in positive_deltas):
        return timedelta(seconds=smallest_gap)

    most_common_gap, _ = Counter(positive_deltas).most_common(1)[0]
    return timedelta(seconds=most_common_gap)


def format_timedelta(step: timedelta) -> str:
    total_seconds = int(step.total_seconds())
    if total_seconds % 86400 == 0:
        return f"{total_seconds // 86400}d"
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}m"
    return f"{total_seconds}s"


def build_missing_timestamps(
    timestamps: list[datetime], step: timedelta
) -> tuple[datetime, datetime, list[datetime]]:
    if step <= timedelta(0):
        raise ValueError("Expected cadence must be greater than zero.")

    unique_timestamps = sorted(set(timestamps))
    start = unique_timestamps[0]
    end = unique_timestamps[-1]
    observed = set(unique_timestamps)

    missing: list[datetime] = []
    current = start
    while current <= end:
        if current not in observed:
            missing.append(current)
        current += step

    return start, end, missing


def build_filename_template(parsed_files: list[ParsedFile]) -> tuple[str, str, str] | None:
    templates = {(parsed.prefix, parsed.suffix, parsed.extension) for parsed in parsed_files}
    if len(templates) != 1:
        return None
    return next(iter(templates))


def main() -> int:
    args = parse_args()
    directory = Path(args.directory)

    try:
        files = load_parquet_files(directory)
        token_regex, datetime_format, parsed_files = detect_datetime_parser(
            files, args.datetime_regex, args.datetime_format
        )
        step = (
            parse_frequency(args.freq)
            if args.freq
            else infer_frequency([parsed.timestamp for parsed in parsed_files])
        )
        earliest, latest, missing = build_missing_timestamps(
            [parsed.timestamp for parsed in parsed_files], step
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    duplicate_count = len(parsed_files) - len({parsed.timestamp for parsed in parsed_files})
    template = build_filename_template(parsed_files)

    print(f"Directory: {directory}")
    print(f"Parquet files scanned: {len(files)}")
    print(f"Datetime regex: {token_regex}")
    print(f"Datetime format: {datetime_format}")
    print(f"Earliest timestamp: {earliest.strftime(datetime_format)}")
    print(f"Latest timestamp: {latest.strftime(datetime_format)}")
    print(f"Expected cadence: {format_timedelta(step)}")
    print(f"Duplicate timestamps: {duplicate_count}")
    print(f"Missing timestamps: {len(missing)}")

    if not missing:
        print("No missing timestamps found in the detected range.")
        return 0

    if template is None:
        print("\nMissing datetimes:")
        for timestamp in missing:
            print(timestamp.strftime(datetime_format))
        return 0

    prefix, suffix, extension = template
    print(f"\nMissing datetimes and expected filenames[{len(missing)}]:")
    for timestamp in missing:
        token = timestamp.strftime(datetime_format)
        print(f"{token}    {prefix}{token}{suffix}{extension}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
