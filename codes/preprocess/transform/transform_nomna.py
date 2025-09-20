import os
import re
import argparse
from concurrent.futures import ThreadPoolExecutor

def normalize_line(line):
    try:
        chinese, vietnamese = line.split('\t')
        chinese = re.sub(r'[^\w\s]', '', chinese)  # Remove special characters from Chinese text
        vietnamese = re.sub(r'[^\w\s]', '', vietnamese).lower().strip()  # Remove special characters, lowercase, and strip Vietnamese text
        vietnamese = re.sub(r'\s+', ' ', vietnamese)  # Ensure single-space separation
        return f"{chinese}\t{vietnamese}"
    except ValueError:
        return None  # Skip lines that don't conform to the expected format

def process_file(file_path):
    normalized_lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            normalized_line = normalize_line(line.strip())
            if normalized_line:
                normalized_lines.append(normalized_line)
    return normalized_lines

def main():
    parser = argparse.ArgumentParser(description="Normalize text corpus.")
    parser.add_argument('--input-dir', required=True, help="Directory containing .txt files")
    parser.add_argument('--output', required=True, help="Target file to save transformed labels")
    parser.add_argument('--threads', type=int, default=16, help="Number of threads to use (default: 16)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output
    num_threads = args.threads

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory '{input_dir}' does not exist or is not a directory.")

    txt_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]

    all_normalized_lines = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(process_file, txt_files)
        for result in results:
            all_normalized_lines.extend(result)

    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write('\n'.join(all_normalized_lines))

if __name__ == "__main__":
    main()
