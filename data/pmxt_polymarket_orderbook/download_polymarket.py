"""Download parquet files from the Polymarket archive."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for lightweight environments

    class tqdm:  # type: ignore[override]
        """Minimal tqdm-compatible fallback."""

        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self) -> "tqdm":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def update(self, n: int = 1) -> None:
            return None

        @staticmethod
        def write(message: str) -> None:
            print(message)


# Configuration
BASE_URL = "https://archive.pmxt.dev/Polymarket"
SAVE_DIR = Path(__file__).parent / "raw"
REQUEST_TIMEOUT = 30
CHUNK_SIZE = 256 * 1024
MAX_RETRIES = 3
MAX_EMPTY_PAGES = 3
DEFAULT_PAGE_WORKERS = 4
DEFAULT_DOWNLOAD_WORKERS = min(16, max(4, (os.cpu_count() or 4) * 2))
DEFAULT_TRUNCATE_WORKERS = max(1, min(4, os.cpu_count() or 1))
USER_AGENT = "onlyPositiveAlpha-polymarket-downloader/2.0"
CURL_BIN = shutil.which("curl")
PAGE_CONTENT_RETRIES = 3
PAGE_RETRY_BACKOFF_SECONDS = 1.5
TEMPORARY_UNAVAILABLE_MARKERS = (
    "service temporarily unavailable",
    "download service is currently down due to high load",
)
LOW_DISK_SPACE_WARNING_BYTES = 5 * 1024**3
TRUNCATE_SCRIPT_PATH = Path(__file__).parent / "truncate_pmxt_by_tokens.py"


def python_ssl_available() -> bool:
    """Return whether this Python runtime can initialize the ssl module."""
    try:
        import ssl  # noqa: F401
    except ImportError:
        return False
    return True


PYTHON_SSL_AVAILABLE = python_ssl_available()


@dataclass(frozen=True)
class DownloadTask:
    """Metadata for a single file download."""

    url: str
    filename: str
    save_path: Path


@dataclass(frozen=True)
class TransferResult:
    """Result for a single file download."""

    task: DownloadTask
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class TruncateResult:
    """Result for a single truncation task."""

    file_path: Path
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class TextResponse:
    """Normalized text response across supported transports."""

    url: str
    text: str


class TransportError(RuntimeError):
    """Raised when the active network transport fails."""


class SessionFactory:
    """Create one configured requests session per worker thread."""

    def __init__(self, pool_size: int) -> None:
        self._pool_size = pool_size
        self._local = threading.local()

    def get(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._build_session()
            self._local.session = session
        return session

    def _build_session(self) -> requests.Session:
        retry = Retry(
            total=MAX_RETRIES,
            connect=MAX_RETRIES,
            read=MAX_RETRIES,
            status=MAX_RETRIES,
            backoff_factor=0.5,
            allowed_methods=frozenset({"GET"}),
            status_forcelist=(429, 500, 502, 503, 504),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=self._pool_size,
            pool_maxsize=self._pool_size,
        )
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session


class RequestsTransport:
    """HTTPS transport backed by pooled requests sessions."""

    def __init__(self, pool_size: int) -> None:
        self._session_factory = SessionFactory(pool_size=pool_size)

    def fetch_text(self, url: str) -> TextResponse:
        session = self._session_factory.get()

        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise TransportError(str(exc)) from exc

        return TextResponse(url=response.url, text=response.text)

    def download(self, url: str, destination: Path) -> None:
        session = self._session_factory.get()

        try:
            with session.get(url, timeout=REQUEST_TIMEOUT, stream=True) as response:
                response.raise_for_status()
                with destination.open("wb") as file_handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            file_handle.write(chunk)
        except requests.RequestException as exc:
            raise TransportError(str(exc)) from exc


class CurlTransport:
    """HTTPS transport backed by the system curl binary."""

    def __init__(self, curl_bin: str) -> None:
        self._curl_bin = curl_bin

    def _build_command(self, url: str, extra_args: list[str] | None = None) -> list[str]:
        command = [
            self._curl_bin,
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            str(MAX_RETRIES),
            "--retry-delay",
            "1",
            "--retry-connrefused",
            "--connect-timeout",
            str(REQUEST_TIMEOUT),
            "--user-agent",
            USER_AGENT,
        ]
        if extra_args:
            command.extend(extra_args)
        command.append(url)
        return command

    def _run(
        self, url: str, extra_args: list[str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            self._build_command(url, extra_args),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error_message = (
                result.stderr.strip()
                or result.stdout.strip()
                or f"curl exited with code {result.returncode}"
            )
            raise TransportError(error_message)
        return result

    def fetch_text(self, url: str) -> TextResponse:
        result = self._run(url, extra_args=["--compressed"])
        return TextResponse(url=url, text=result.stdout)

    def download(self, url: str, destination: Path) -> None:
        self._run(url, extra_args=["--output", str(destination)])


ArchiveTransport = RequestsTransport | CurlTransport


def build_transport(pool_size: int) -> ArchiveTransport:
    """Create a transport that works in the current runtime."""
    if PYTHON_SSL_AVAILABLE:
        return RequestsTransport(pool_size=pool_size)

    if CURL_BIN is not None:
        return CurlTransport(curl_bin=CURL_BIN)

    raise RuntimeError("Python SSL is unavailable and no 'curl' executable was found on PATH.")


def describe_transport() -> str:
    """Return a human-readable description of the active transport."""
    if PYTHON_SSL_AVAILABLE:
        return "requests"

    if CURL_BIN is not None:
        return f"curl fallback ({CURL_BIN}; Python SSL unavailable)"

    return "unavailable"


def is_temporary_unavailable_page(page_text: str) -> bool:
    """Return whether the archive responded with a transient service page."""
    normalized_text = page_text.lower()
    return any(marker in normalized_text for marker in TEMPORARY_UNAVAILABLE_MARKERS)


def format_bytes(num_bytes: int) -> str:
    """Format a byte count for logs."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def free_space_bytes(path: Path) -> int:
    """Return free bytes on the filesystem containing path."""
    target = path if path.exists() else path.parent
    return shutil.disk_usage(target).free


def annotate_write_failure(error: Exception, destination_dir: Path) -> str:
    """Add actionable context to local write failures."""
    message = str(error)
    normalized = message.lower()
    if (
        "failure writing output to destination" not in normalized
        and "no space left on device" not in normalized
    ):
        return message

    free_bytes = free_space_bytes(destination_dir)
    return (
        f"{message} "
        f"(local filesystem write failed; free space in {destination_dir} is "
        f"{format_bytes(free_bytes)})"
    )


class ParquetLinkParser(HTMLParser):
    """Collect .parquet links from simple archive index pages."""

    def __init__(self) -> None:
        super().__init__()
        self.links: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return

        for attr_name, attr_value in attrs:
            if attr_name == "href" and attr_value and attr_value.endswith(".parquet"):
                self.links.add(attr_value)
                return


def parse_iso_date(value: str) -> date:
    """Validate YYYY-MM-DD CLI input."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected format: YYYY-MM-DD."
        ) from exc


def parse_positive_int(value: str) -> int:
    """Validate positive integer CLI input."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got '{value}'.") from exc

    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got '{value}'.")

    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date", type=parse_iso_date, help="Only download files on/after date."
    )
    parser.add_argument("--base-url", default=BASE_URL, help="Archive base URL to scan.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=SAVE_DIR,
        help="Directory where parquet files are stored.",
    )
    parser.add_argument(
        "--page-workers",
        type=parse_positive_int,
        default=DEFAULT_PAGE_WORKERS,
        help="Concurrent page fetch workers.",
    )
    parser.add_argument(
        "--download-workers",
        type=parse_positive_int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        help="Concurrent file download workers.",
    )
    parser.add_argument(
        "--truncate-workers",
        type=parse_positive_int,
        default=DEFAULT_TRUNCATE_WORKERS,
        help="Concurrent truncation workers.",
    )
    parser.add_argument(
        "--max-empty-pages",
        type=parse_positive_int,
        default=MAX_EMPTY_PAGES,
        help="Stop after this many consecutive pages without parquet links.",
    )
    parser.add_argument(
        "--skip-truncate",
        action="store_true",
        help="Download files without invoking truncate_pmxt_by_tokens.py.",
    )
    return parser.parse_args(argv)


def extract_date_from_filename(filename: str) -> date | None:
    """Extract YYYY-MM-DD date from a parquet filename."""
    if not filename.endswith(".parquet"):
        return None

    stem = filename[:-8]
    last_token = stem.split("_")[-1]
    date_token = last_token.split("T", 1)[0]
    try:
        return date.fromisoformat(date_token)
    except ValueError:
        return None


def fetch_page_links(
    page_num: int,
    base_url: str,
    transport: ArchiveTransport,
) -> tuple[int, set[str]]:
    """Fetch one archive page and return parquet links."""
    url = f"{base_url}?page={page_num}" if page_num > 1 else base_url
    last_error: str | None = None

    for attempt in range(PAGE_CONTENT_RETRIES + 1):
        response = transport.fetch_text(url)

        parser = ParquetLinkParser()
        parser.feed(response.text)
        parquet_links = {urljoin(response.url, href) for href in parser.links}
        if parquet_links:
            return page_num, parquet_links

        if not is_temporary_unavailable_page(response.text):
            return page_num, set()

        last_error = "archive page reported temporary unavailability"
        if attempt == PAGE_CONTENT_RETRIES:
            break

        time.sleep(PAGE_RETRY_BACKOFF_SECONDS * (attempt + 1))

    raise TransportError(last_error or "archive page did not return parquet links")


def collect_parquet_links(base_url: str, page_workers: int, max_empty_pages: int) -> set[str]:
    """Scan archive pages in concurrent batches while preserving empty-page stop logic."""
    all_links: set[str] = set()
    page_num = 1
    consecutive_empty_pages = 0
    transport = build_transport(pool_size=max(8, page_workers * 2))

    with futures.ThreadPoolExecutor(
        max_workers=page_workers,
        thread_name_prefix="page-fetch",
    ) as executor:
        while consecutive_empty_pages < max_empty_pages:
            page_batch = list(range(page_num, page_num + page_workers))
            print(f"Scanning pages {page_batch[0]}-{page_batch[-1]}...")

            future_to_page = {
                executor.submit(fetch_page_links, page, base_url, transport): page
                for page in page_batch
            }
            batch_results: dict[int, set[str]] = {}

            for future in futures.as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    _, links = future.result()
                except TransportError as exc:
                    print(f"  Page {page}: request failed after retries: {exc}")
                    links = set()
                except Exception as exc:
                    print(f"  Page {page}: unexpected error: {exc}")
                    links = set()
                batch_results[page] = links

            for page in page_batch:
                page_links = batch_results[page]
                print(f"  Page {page}: {len(page_links)} parquet files")
                if page_links:
                    consecutive_empty_pages = 0
                    all_links.update(page_links)
                    continue

                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_empty_pages:
                    break

            page_num += page_workers

    return all_links


def filter_links_by_start_date(
    all_links: set[str], start_date: date | None
) -> tuple[set[str], int]:
    """Apply optional start-date filtering."""
    if start_date is None:
        return all_links, 0

    filtered_links: set[str] = set()
    skipped_before_start = 0

    for url in all_links:
        filename = urlparse(url).path.split("/")[-1]
        file_date = extract_date_from_filename(filename)
        if file_date is None or file_date >= start_date:
            filtered_links.add(url)
        else:
            skipped_before_start += 1

    return filtered_links, skipped_before_start


def build_download_tasks(
    all_links: set[str], save_dir: Path
) -> tuple[list[DownloadTask], int, int]:
    """Create concrete download tasks and count skipped files."""
    tasks: list[DownloadTask] = []
    skipped_existing = 0
    skipped_invalid = 0
    existing_filenames = {path.name for path in save_dir.glob("*.parquet")}

    for url in sorted(all_links):
        filename = urlparse(url).path.split("/")[-1]
        if not filename or not filename.endswith(".parquet"):
            skipped_invalid += 1
            continue

        save_path = save_dir / filename
        if filename in existing_filenames or save_path.exists():
            skipped_existing += 1
            continue

        tasks.append(DownloadTask(url=url, filename=filename, save_path=save_path))

    return tasks, skipped_existing, skipped_invalid


def download_file(task: DownloadTask, transport: ArchiveTransport) -> TransferResult:
    """Download a single file to an atomic temporary path."""
    temp_path = task.save_path.with_suffix(f"{task.save_path.suffix}.part")

    try:
        if temp_path.exists():
            temp_path.unlink()

        transport.download(task.url, temp_path)
        temp_path.replace(task.save_path)
        return TransferResult(task=task, success=True)
    except (OSError, TransportError) as exc:
        if temp_path.exists():
            temp_path.unlink()
        error_message = annotate_write_failure(exc, task.save_path.parent)
        return TransferResult(task=task, success=False, error=error_message)


def run_truncate_on_file(file_path: Path) -> TruncateResult:
    """Run truncate_pmxt_by_tokens.py for one downloaded file."""

    script_path = TRUNCATE_SCRIPT_PATH

    if not script_path.exists():
        return TruncateResult(
            file_path=file_path,
            success=False,
            error=f"Truncate script not found at {script_path}",
        )

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--file", str(file_path)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return TruncateResult(file_path=file_path, success=False, error=str(exc))

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        error_message = stderr or stdout or f"Truncation exited with code {result.returncode}"
        return TruncateResult(file_path=file_path, success=False, error=error_message)

    return TruncateResult(file_path=file_path, success=True)


def process_downloads(
    tasks: list[DownloadTask],
    download_workers: int,
    truncate_workers: int,
    skip_truncate: bool,
) -> tuple[int, int, int]:
    """Download files concurrently and pipeline truncation work."""
    if not tasks:
        return 0, 0, 0

    download_transport = build_transport(pool_size=max(8, download_workers * 2))
    downloaded_count = 0
    failed_count = 0
    truncate_failed_count = 0
    truncate_futures: list[futures.Future[TruncateResult]] = []

    with futures.ThreadPoolExecutor(
        max_workers=download_workers,
        thread_name_prefix="download",
    ) as download_executor:
        truncate_executor: futures.ThreadPoolExecutor | None = None
        if not skip_truncate:
            truncate_executor = futures.ThreadPoolExecutor(
                max_workers=truncate_workers,
                thread_name_prefix="truncate",
            )

        try:
            future_to_task = {
                download_executor.submit(download_file, task, download_transport): task
                for task in tasks
            }

            with tqdm(total=len(tasks), unit="file", desc="Downloading") as progress:
                for future in futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        failed_count += 1
                        progress.update(1)
                        tqdm.write(f"Download failed for {task.filename}: {exc}")
                        continue

                    progress.update(1)

                    if result.success:
                        downloaded_count += 1
                        if truncate_executor is not None:
                            truncate_futures.append(
                                truncate_executor.submit(
                                    run_truncate_on_file, result.task.save_path
                                )
                            )
                        continue

                    failed_count += 1
                    tqdm.write(f"Download failed for {result.task.filename}: {result.error}")
        finally:
            if truncate_executor is not None:
                truncate_executor.shutdown(wait=True)

    for truncate_future in truncate_futures:
        truncate_result = truncate_future.result()
        if not truncate_result.success:
            truncate_failed_count += 1
            print(
                f"Truncation failed for {truncate_result.file_path.name}: {truncate_result.error}"
            )

    return downloaded_count, failed_count, truncate_failed_count


def main(argv: list[str] | None = None) -> int:
    """Download all parquet files from the Polymarket archive."""
    args = parse_args(argv)
    try:
        transport_description = describe_transport()
        build_transport(pool_size=max(args.page_workers, args.download_workers))
    except RuntimeError as exc:
        print(f"Transport configuration error: {exc}")
        return 1

    args.save_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = free_space_bytes(args.save_dir)
    if not args.skip_truncate and not TRUNCATE_SCRIPT_PATH.exists():
        print(f"Truncation disabled: script not found at {TRUNCATE_SCRIPT_PATH}")
        args.skip_truncate = True

    print(f"Saving files to: {args.save_dir}")
    if args.start_date is not None:
        print(f"Filtering to files dated on or after: {args.start_date.isoformat()}")
    print(
        "Workers:"
        f" pages={args.page_workers},"
        f" downloads={args.download_workers},"
        f" truncation={'disabled' if args.skip_truncate else args.truncate_workers}"
    )
    print(f"Transport: {transport_description}")
    print(f"Free space: {format_bytes(free_bytes)}")
    if free_bytes < LOW_DISK_SPACE_WARNING_BYTES:
        print(
            "Warning: low free space detected. Large parquet downloads may fail "
            "until disk space is freed."
        )
    print()

    all_links = collect_parquet_links(
        base_url=args.base_url,
        page_workers=args.page_workers,
        max_empty_pages=args.max_empty_pages,
    )
    if not all_links:
        print("No parquet files found on any pages.")
        return 0

    all_links, skipped_before_start = filter_links_by_start_date(all_links, args.start_date)
    if args.start_date is not None:
        print(f"Skipped {skipped_before_start} files before start date.\n")

    if not all_links:
        print("No parquet files matched the requested date filter.")
        return 0

    tasks, skipped_existing, skipped_invalid = build_download_tasks(all_links, args.save_dir)

    print(f"Total unique parquet files found: {len(all_links)}")
    print(f"Already present: {skipped_existing}")
    print(f"Invalid links:    {skipped_invalid}")
    print(f"Queued downloads: {len(tasks)}\n")

    downloaded_count, failed_count, truncate_failed_count = process_downloads(
        tasks=tasks,
        download_workers=args.download_workers,
        truncate_workers=args.truncate_workers,
        skip_truncate=args.skip_truncate,
    )

    print("\n" + "=" * 60)
    print("Download Summary:")
    print(f"  Downloaded:        {downloaded_count}")
    print(f"  Skipped existing:  {skipped_existing}")
    print(f"  Skipped invalid:   {skipped_invalid}")
    print(f"  Failed downloads:  {failed_count}")
    if args.skip_truncate:
        print("  Truncation:        skipped")
    else:
        print(f"  Failed truncation: {truncate_failed_count}")
    print(f"  Total discovered:  {len(all_links)}")
    print("=" * 60)
    return 0 if failed_count == 0 and truncate_failed_count == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        raise SystemExit(1)
