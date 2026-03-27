"""
Download all parquet files from Polymarket archive.

across all pages, starting from page 1.
"""

import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configuration
BASE_URL = "https://archive.pmxt.dev/Polymarket"
SAVE_DIR = Path(__file__).parent.parent / "data" / "cached" / "pmxt"
REQUEST_TIMEOUT = 30
CHUNK_SIZE = 8192
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def extract_date_from_filename(filename: str) -> date | None:
    """Extract YYYY-MM-DD date from a parquet filename.

    Expects the date to be the last underscore-separated token before `.parquet`.
    """
    if not filename.endswith(".parquet"):
        return None

    stem = filename[:-8]  # remove ".parquet"
    last_token = stem.split("_")[-1]
    date_token = last_token.split("T", 1)[0]
    try:
        return date.fromisoformat(date_token)
    except ValueError:
        return None


def run_truncate_on_file(file_path: Path) -> None:
    """Run truncate_pmxt_by_tokens.py script on a specific downloaded file."""
    script_path = Path(__file__).parent / "truncate_pmxt_by_tokens.py"

    if not script_path.exists():
        print(f"Warning: truncate script not found at {script_path}")
        return

    print(f">>> Truncating {file_path.name}...")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--file", str(file_path)],
            cwd=Path(__file__).parent.parent,
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"Warning: truncation script exited with code {result.returncode}")
    except Exception as e:
        print(f"Error running truncation script: {e}")
    print(">>> Truncation complete\n")


def parse_start_date_arg() -> date | None:
    """Parse optional --start-date YYYY-MM-DD from command-line arguments."""
    args = sys.argv[1:]
    if not args:
        return None

    if len(args) != 2 or args[0] != "--start-date":
        print("Usage: python scripts/download_polymarket.py [--start-date YYYY-MM-DD]")
        sys.exit(2)

    try:
        return date.fromisoformat(args[1])
    except ValueError:
        print(f"Invalid start date '{args[1]}'. Expected format: YYYY-MM-DD")
        sys.exit(2)


def get_parquet_links(page_num: int) -> set[str]:
    """
    Fetch a page and extract all parquet file links with retry logic.

    Args:
        page_num: Page number to fetch

    Returns:
        Set of absolute URLs to parquet files
    """
    # Page 1 has no query parameter
    url = f"{BASE_URL}?page={page_num}" if page_num > 1 else BASE_URL

    print(f"Fetching page {page_num}: {url}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt < MAX_RETRIES:
                print(f"  Error fetching page {page_num} (attempt {attempt}/{MAX_RETRIES}): {e}")
                print(f"  Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Error fetching page {page_num} after {MAX_RETRIES} attempts: {e}")
                return set()

    soup = BeautifulSoup(response.content, "html.parser")

    # Find all links that end with .parquet
    parquet_links = set()
    for link in soup.find_all("a", href=True):
        href = str(link.get("href", ""))
        if href.endswith(".parquet"):
            # Convert to absolute URL if relative
            absolute_url = urljoin(response.url, href)
            parquet_links.add(absolute_url)

    print(f"Found {len(parquet_links)} parquet files on page {page_num}")
    return parquet_links


def download_file(url: str, save_path: Path) -> bool:
    """
    Download a file from URL to the specified path with retry logic.

    Args:
        url: URL of the file to download
        save_path: Path where to save the file

    Returns:
        True if successful, False otherwise
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with save_path.open("wb") as f:
                if total_size > 0:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=save_path.name,
                        leave=False,
                    ) as pbar:
                        for chunk in response.iter_content(CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"Error downloading {save_path.name} (attempt {attempt}/{MAX_RETRIES}): {e}")
                print(f"Retrying in {RETRY_DELAY} seconds...")
                # Clean up partial file if it exists
                if save_path.exists():
                    save_path.unlink()
                time.sleep(RETRY_DELAY)
            else:
                print(f"Error downloading {save_path.name} after {MAX_RETRIES} attempts: {e}")
                # Clean up partial file
                if save_path.exists():
                    save_path.unlink()
                return False
        else:
            return True

    return False


def main() -> None:
    """Download all parquet files from the Polymarket archive."""
    start_date = parse_start_date_arg()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving files to: {SAVE_DIR}\n")
    if start_date is not None:
        print(f"Filtering to files dated on or after: {start_date.isoformat()}\n")

    all_links: set[str] = set()

    page_num = 1
    consecutive_empty_pages = 0
    max_empty_pages = 3  # Stop after 3 consecutive empty pages

    while consecutive_empty_pages < max_empty_pages:
        page_links = get_parquet_links(page_num)

        if not page_links:
            consecutive_empty_pages += 1
            print(
                "No parquet files found. "
                f"Empty page count: {consecutive_empty_pages}/{max_empty_pages}\n"
            )
        else:
            consecutive_empty_pages = 0
            all_links.update(page_links)

        page_num += 1

    if not all_links:
        print("No parquet files found on any pages.")
        return

    if start_date is not None:
        filtered_links: set[str] = set()
        skipped_before_start = 0

        for url in all_links:
            filename = urlparse(url).path.split("/")[-1]
            file_date = extract_date_from_filename(filename)
            if file_date is None or file_date >= start_date:
                filtered_links.add(url)
            else:
                skipped_before_start += 1

        all_links = filtered_links
        print(f"Skipped {skipped_before_start} files before start date.\n")

    if not all_links:
        print("No parquet files matched the requested date filter.")
        return

    print(f"\nTotal unique parquet files found: {len(all_links)}\n")

    # Download all files
    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    for i, url in enumerate(sorted(all_links), 1):
        filename = urlparse(url).path.split("/")[-1]

        if not filename or not filename.endswith(".parquet"):
            print(f"[{i}/{len(all_links)}] Skipping invalid filename: {url}")
            skipped_count += 1
            continue

        save_path = SAVE_DIR / filename

        if save_path.exists():
            print(f"[{i}/{len(all_links)}] Already exists: {filename}")
            skipped_count += 1
            continue

        print(f"[{i}/{len(all_links)}] Downloading: {filename}")
        if download_file(url, save_path):
            downloaded_count += 1

            # Truncate this specific file immediately after download
            run_truncate_on_file(save_path)
        else:
            failed_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary:")
    print(f"  Downloaded: {downloaded_count}")
    print(f"  Skipped:    {skipped_count}")
    print(f"  Failed:     {failed_count}")
    print(f"  Total:      {len(all_links)}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
