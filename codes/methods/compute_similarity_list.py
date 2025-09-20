import csv
import argparse
from concurrent.futures import ProcessPoolExecutor
from Levenshtein import distance as levenshtein_distance
import json
from tqdm import tqdm
import os
from bisect import bisect_left, bisect_right


def compute_similarity(idx, all_chars, all_lengths, threshold):
    char, strokes = all_chars[idx]
    len_strokes = all_lengths[idx]
    similar_pairs = []

    # Approximate bounds for window: derived from abs(len1 - len2) < threshold * (len1 + len2) / 2
    # Simplifying, lower ~ len_strokes * (1 - threshold) / (1 + threshold), but use heuristics for speed
    factor = 1 / (1 + threshold)  # Conservative approximation
    lower_bound = int(len_strokes * factor)
    upper_bound = int(len_strokes / factor)
    start = bisect_left(all_lengths, lower_bound)
    end = bisect_right(all_lengths, upper_bound)

    for j in range(start, end):
        other_char, other_strokes = all_chars[j]
        if other_char == char:
            continue
        len_other = all_lengths[j]
        max_len = threshold * (len_strokes + len_other) / 2  # Use threshold properly (original hardcoded 0.5)
        if abs(len_strokes - len_other) >= max_len:
            continue  # Skip impossible pairs
        edit_distance = levenshtein_distance(strokes, other_strokes)
        if edit_distance < max_len:
            similar_pairs.append((edit_distance, other_char))
    
    # Sort by edit_distance ascending (most similar first)
    similar_pairs.sort(key=lambda x: x[0])
    similar_chars = [pair[1] for pair in similar_pairs][:20]
    
    return char, similar_chars


def process_file(input_path, output_path, num_processes, threshold):
    with open(input_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        all_chars = [(row['ID'], row['Strokes']) for row in reader if row['Strokes']]

    all_chars.sort(key=lambda x: len(x[1]))  # Sort by stroke length for windowing
    all_lengths = [len(s) for _, s in all_chars]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results_iter = executor.map(
            compute_similarity,
            range(len(all_chars)),
            [all_chars] * len(all_chars),
            [all_lengths] * len(all_chars),
            [threshold] * len(all_chars)
        )
        results = list(tqdm(results_iter, total=len(all_chars), desc="Computing similarities"))

    with open(output_path, mode='w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=":")
        writer.writerow(['Char', 'Candidates'])
        for char, candidates in results:
            writer.writerow([char, "".join(candidates)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute visual similarity of characters in a CSV file.")
    parser.add_argument('--input', required=True, help="Path to the input CSV file.")
    parser.add_argument('--output', required=True, help="Path to the output CSV file.")
    parser.add_argument('--processes', type=int, default=os.cpu_count(), help="Number of processes to use.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Similarity threshold (default: 0.5).")
    args = parser.parse_args()

    process_file(args.input, args.output, args.processes, args.threshold)