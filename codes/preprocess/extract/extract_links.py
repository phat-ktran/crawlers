#!/usr/bin/env python3
"""
CLI program to extract poem links from HTML files with ThreadPoolExecutor for performance.

This script searches for div elements with class 'list-item' within
'page-content-main' > 'sticky-top' structure and extracts poem titles
and their href links.

Usage:
    python extract_links.py --input-dir /path/to/html/files --output output.csv
    python extract_links.py --input-dir /path/to/html/files --output output.csv --links links.txt
"""

import argparse
import csv
import sys
from pathlib import Path
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from bs4 import BeautifulSoup


# Thread-safe counter for progress tracking
class Counter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self):
        with self._lock:
            return self._value


def load_links_file(links_file_path):
    """
    Load the links file and return a dictionary mapping ID to [URL, status].

    Args:
        links_file_path (str): Path to the links file

    Returns:
        dict: Dictionary mapping ID to [URL, status]
    """
    links_data = {}

    try:
        with open(links_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) != 3:
                    print(
                        f"Warning: Invalid format at line {line_num} in {links_file_path}: {line}"
                    )
                    continue

                id_part, url_part, status_part = parts
                try:
                    status = int(status_part)
                    links_data[id_part] = [url_part, status]
                except ValueError:
                    print(
                        f"Warning: Invalid status at line {line_num} in {links_file_path}: {status_part}"
                    )
                    continue

    except Exception as e:
        print(f"Error reading links file {links_file_path}: {e}")

    return links_data


def save_links_file(links_data, links_file_path):
    """
    Save the updated links data back to the file.

    Args:
        links_data (dict): Dictionary mapping ID to [URL, status]
        links_file_path (str): Path to the links file
    """
    try:
        with open(links_file_path, "w", encoding="utf-8") as f:
            for id_part, (url_part, status) in links_data.items():
                f.write(f"{id_part},{url_part},{status}\n")
    except Exception as e:
        print(f"Error writing links file {links_file_path}: {e}")


def extract_id_from_filename(filename):
    """
    Extract ID from HTML filename.

    Args:
        filename (str): HTML filename like "27_51_1.html"

    Returns:
        str: ID part like "27_51_1"
    """
    # Remove .html or .htm extension
    name = filename.lower()
    if name.endswith(".html"):
        return filename[:-5]
    elif name.endswith(".htm"):
        return filename[:-4]
    else:
        return filename


def extract_links_from_html(html_file_path):
    """
    Extract poem links from a single HTML file.

    Args:
        html_file_path (Path): Path to the HTML file

    Returns:
        tuple: (file_name, file_id, links, error)
    """
    links = []
    file_name = html_file_path.name
    file_id = extract_id_from_filename(file_name)

    try:
        with open(html_file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return file_name, file_id, [], f"Could not read file: {e}"

    try:
        soup = BeautifulSoup(content, "html.parser")

        # Find div with class "page-content-main"
        page_content_main = soup.find("div", class_="page-content-main")
        if not page_content_main:
            return file_name, file_id, [], "No 'page-content-main' div found"

        # Find child div with class "sticky-top"
        sticky_top = page_content_main.find("div", class_="sticky-top")
        if not sticky_top:
            return file_name, file_id, [], "No 'sticky-top' div found"

        # Find all child divs with class "list-item"
        list_items = sticky_top.find_all("div", class_="list-item")
        if not list_items:
            return file_name, file_id, [], "No 'list-item' divs found"

        for list_item in list_items:
            # Find h4 with class "list-item-header"
            header = list_item.find("h4", class_="list-item-header")
            if not header:
                continue

            # Find the anchor tag within the h4
            anchor = header.find("a")
            if not anchor:
                continue

            # Extract title (text content) and href
            title = anchor.get_text(strip=True)
            href = anchor.get("href")

            if title and href:
                # URL decode the title for use as ID
                title_decoded = unquote(title)
                # Clean up title to make it a valid ID (remove special characters, spaces)
                title_id = "".join(
                    c for c in title_decoded if c.isalnum() or c in "-_"
                ).strip()

                # Form complete URL
                full_url = f"https://www.thivien.net{href}"

                links.append((title_id, full_url))

    except Exception as e:
        return file_name, file_id, [], f"Error parsing HTML: {e}"

    return file_name, file_id, links, None


def process_files_parallel(html_files, max_workers=None, links_data=None):
    """
    Process HTML files in parallel using ThreadPoolExecutor.

    Args:
        html_files (list): List of Path objects for HTML files
        max_workers (int): Maximum number of worker threads
        links_data (dict): Dictionary mapping ID to [URL, status] for tracking

    Returns:
        tuple: (all_links, total_processed, total_errors, updated_links_data)
    """
    all_links = []
    counter = Counter()
    errors = []
    total_files = len(html_files)

    # Thread-safe dictionary for updating links data
    links_lock = threading.Lock()

    # Use default number of workers if not specified
    if max_workers is None:
        max_workers = min(32, (len(html_files) or 4) + 4)

    print(f"Processing {total_files} files with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(extract_links_from_html, html_file): html_file
            for html_file in html_files
        }

        # Process completed tasks
        for future in as_completed(future_to_file):
            html_file = future_to_file[future]
            processed_count = counter.increment()

            try:
                file_name, file_id, links, error = future.result()

                if error:
                    errors.append(f"{file_name}: {error}")
                    print(f"  [{processed_count}/{total_files}] {file_name}: {error}")

                    # Update links_data if available - set status to 0 for files with errors
                    if links_data is not None:
                        with links_lock:
                            if file_id in links_data:
                                links_data[file_id][1] = 0
                else:
                    if links:
                        all_links.extend(links)
                        print(
                            f"  [{processed_count}/{total_files}] {file_name}: {len(links)} links"
                        )
                    else:
                        print(
                            f"  [{processed_count}/{total_files}] {file_name}: No links found"
                        )

                        # Update links_data if available - set status to 0 for files with no links
                        if links_data is not None:
                            with links_lock:
                                if file_id in links_data:
                                    links_data[file_id][1] = 0

            except Exception as e:
                errors.append(f"{html_file.name}: Unexpected error: {e}")
                print(
                    f"  [{processed_count}/{total_files}] {html_file.name}: Unexpected error: {e}"
                )

                # Update links_data if available - set status to 0 for files with unexpected errors
                if links_data is not None:
                    file_id = extract_id_from_filename(html_file.name)
                    with links_lock:
                        if file_id in links_data:
                            links_data[file_id][1] = 0

    return all_links, counter.value, errors


def main():
    """Main function to handle CLI arguments and process files."""
    parser = argparse.ArgumentParser(
        description="Extract poem links from HTML files with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_links.py --input-dir ./html_files --output links.csv
    python extract_links.py --input-dir /path/to/htmls --output /path/to/output.csv --workers 16
    python extract_links.py --input-dir ./htmls --output output.csv --links links.txt --workers 8
        """,
    )

    parser.add_argument(
        "--input-dir", required=True, help="Directory containing HTML files to process"
    )

    parser.add_argument("--output", required=True, help="Output CSV file path")

    parser.add_argument(
        "--links",
        help="Links file to update (format: <ID>,<URL>,<status>). Files with no links will have status changed from 1 to 0.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads (default: auto-detect based on CPU count)",
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        sys.exit(1)

    # Load links data if provided
    links_data = None
    if args.links:
        links_file_path = Path(args.links)
        if not links_file_path.exists():
            print(f"Error: Links file '{links_file_path}' does not exist")
            sys.exit(1)

        print(f"Loading links file: {links_file_path}")
        links_data = load_links_file(links_file_path)
        print(f"Loaded {len(links_data)} entries from links file")

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all HTML files in the input directory
    html_files = list(input_dir.glob("*.html")) + list(input_dir.glob("*.htm"))

    if not html_files:
        print(f"Warning: No HTML files found in '{input_dir}'")
        sys.exit(0)

    print(f"Found {len(html_files)} HTML files to process")

    # Process files in parallel
    all_links, processed_files, errors = process_files_parallel(
        html_files, args.workers, links_data
    )

    # Print errors if any
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    # Write results to CSV file
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Deduplicate by title_id
            seen_title_ids = set()
            for title_id, url in all_links:
                if title_id not in seen_title_ids:
                    writer.writerow([title_id, url, 0])
                    seen_title_ids.add(title_id)

        print(f"\nResults written to: {output_path}")
        print(f"Total files processed: {processed_files}")
        print(f"Total links extracted: {len(all_links)}")
        print(f"Total errors: {len(errors)}")

    except Exception as e:
        print(f"Error writing to output file '{output_path}': {e}")
        sys.exit(1)

    # Save updated links file if provided
    if args.links and links_data is not None:
        try:
            save_links_file(links_data, args.links)

            # Count how many entries were updated to status 0
            updated_count = sum(1 for url, status in links_data.values() if status == 0)
            print(f"Updated links file: {args.links}")
            print(f"Entries with status changed to 0: {updated_count}")

        except Exception as e:
            print(f"Error updating links file '{args.links}': {e}")


if __name__ == "__main__":
    main()
