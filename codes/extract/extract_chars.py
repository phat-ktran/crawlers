import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


def parse_html(file_path: Path) -> dict:
    """Parse a single HTML file and return extracted info."""
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Extract the character
    character = ""
    word_div = soup.find("div", class_="hvres-word han")
    if word_div:
        character = word_div.text.strip()

    # Extract strokes from first <div class="hvres-meaning">
    strokes = ""
    radicals = ""
    meaning_divs = soup.find_all("div", class_="hvres-meaning")
    if meaning_divs:
        first_meaning = meaning_divs[0]
        content = list(first_meaning.stripped_strings)
        content = [s.strip() for s in content]
        try:
            stroke_idx = content.index("Nét bút:")
            if stroke_idx >= 0 and stroke_idx + 1 < len(content):
                strokes = content[stroke_idx + 1]
        except ValueError:
            pass

        try:
            radical_idx = content.index("Hình thái:")
            if radical_idx >= 0 and radical_idx + 1 < len(content):
                radicals = content[radical_idx + 1]
        except ValueError:
            pass

    return {
        "ID": file_path.stem,
        "Character": character,
        "Strokes": strokes,
        "Radicals": radicals,
    }


def process_chunk(file_paths, threads_per_process: int):
    """Process a chunk of files using ThreadPoolExecutor inside one process."""
    with ThreadPoolExecutor(max_workers=threads_per_process) as executor:
        results = list(executor.map(parse_html, file_paths))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Process HTML files with ProcessPool + ThreadPool executors."
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing HTML files."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save outputs."
    )
    parser.add_argument(
        "--max-processes", type=int, default=4, help="Number of processes."
    )
    parser.add_argument(
        "--threads-per-process", type=int, default=8, help="Threads per process."
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    xlsx_dir = output_dir / "xlsx"
    json_dir = output_dir / "json"
    os.makedirs(xlsx_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    html_files = list(input_dir.glob("*.html"))

    # Split into chunks for processes
    chunk_size = max(1, len(html_files) // args.max_processes)
    chunks = [
        html_files[i : i + chunk_size] for i in range(0, len(html_files), chunk_size)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=args.max_processes) as executor:
        futures = [
            executor.submit(process_chunk, chunk, args.threads_per_process)
            for chunk in chunks
        ]
        for f in futures:
            results.extend(f.result())

    # Save to Excel
    df = pd.DataFrame(results)
    xlsx_path = xlsx_dir / "output.xlsx"
    df.to_excel(xlsx_path, index=False, columns=["ID", "Strokes", "Radicals"])

    # Save JSON
    counter = 0
    empty = 0
    for result in results:
        character = result["Character"]
        strokes = result["Strokes"]
        if character and strokes:
            counter += 1
            json_data = {"character": character, "strokes": strokes}
            json_path = json_dir / f"{character}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        else:
            empty += 1

    print(f"✅ Total entries saved as JSON: {counter}")
    print(f"✅ Total entries empty: {empty}")


if __name__ == "__main__":
    main()
