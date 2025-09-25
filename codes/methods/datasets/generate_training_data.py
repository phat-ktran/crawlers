import argparse
import csv
import os
import random
import sys
from collections import defaultdict
import heapq

def main():
    parser = argparse.ArgumentParser(description="Generate character-level noisy data for Sino-Nom OCR correction")
    parser.add_argument('--input-dir', type=str, required=True, help="Input directory containing CLC.txt, NomNaNMT.txt, ThiVien.txt files")
    parser.add_argument('--ratio', type=str, required=True, help="Error ratio string in format a:b:c (substitution:deletion:insertion)")
    parser.add_argument('--similarity', type=str, required=True, help="Similarity CSV with format Char:Candidates (sep=':')")
    parser.add_argument('--stroke', type=str, required=True, help="Stroke CSV with columns ID,Strokes,Radicals (sep=',')")
    parser.add_argument('--num_val', type=int, default=10000, help="Number of validation lines (default: 10000)")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory for train/val files")
    parser.add_argument('--seed', type=int, default=17, help="Random seed (default: 17)")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    
    # Load similarity dictionary
    similarity_dict = {}
    try:
        with open(args.similarity, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':', 1)
                if len(parts) < 2: 
                    continue
                char = parts[0].strip()
                candidates = list(parts[1])
                if candidates:
                    similarity_dict[char] = candidates
    except Exception as e:
        print(f"Error loading similarity file: {e}", file=sys.stderr)
        sys.exit(1)

    # Load stroke dictionary
    stroke_dict = {}
    try:
        with open(args.stroke, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                char = row[0].strip()
                strokes = row[1].strip()
                if strokes.isdigit():
                    stroke_dict[char] = int(strokes)
    except Exception as e:
        print(f"Error loading stroke file: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse ratio
    ratio_parts = args.ratio.split(':')
    if len(ratio_parts) != 3:
        print("Error: Ratio must be exactly 3 integers (a:b:c)", file=sys.stderr)
        sys.exit(1)
    try:
        sub_weight = int(ratio_parts[0])
        del_weight = int(ratio_parts[1])
        ins_weight = int(ratio_parts[2])
    except ValueError:
        print("Error: Ratio parts must be integers", file=sys.stderr)
        sys.exit(1)
    
    # Threshold for deletion (stroke count <= 2 considered "thin strokes")
    THRESHOLD = 2

    # Read input files from directory
    file_data = {}
    required_files = ['CLC.txt', 'NomNaNMT.txt', 'ThiVien.txt']
    
    for filename in required_files:
        filepath = os.path.join(args.input_dir, filename)
        if not os.path.exists(filepath):
            print(f"Error: Required file {filename} not found in {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        
        file_data[filename] = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    sino = parts[0].strip()
                    viet = parts[1].strip()
                    file_data[filename].append((sino, viet))
            print(f"Loaded {len(file_data[filename])} lines from {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    if not any(file_data.values()):
        print("Error: No valid lines found in any input file", file=sys.stderr)
        sys.exit(1)

    # Calculate overall vocabulary across all files
    overall_vocab = set()
    for filename, lines in file_data.items():
        for sino, _ in lines:
            for char in sino.strip():
                overall_vocab.add(char)
    
    print(f"Overall vocabulary size: {len(overall_vocab)} unique characters")

    # Shuffle data within each file
    for filename in file_data:
        random.shuffle(file_data[filename])

    # Select pretraining data
    PRETRAINING_TARGET = 100000
    pretraining_lines = []
    
    # First, take all samples from CLC.txt
    clc_lines = file_data['CLC.txt']
    pretraining_lines.extend(clc_lines)
    print(f"Added all {len(clc_lines)} lines from CLC.txt to pretraining")
    
    # Calculate remaining needed samples
    remaining_needed = PRETRAINING_TARGET - len(clc_lines)
    
    if remaining_needed > 0:
        # Combine ThiVien and NomNaNMT for selection
        combined_pool = []
        for sino, viet in file_data['ThiVien.txt']:
            combined_pool.append((sino, viet, 'ThiVien'))
        for sino, viet in file_data['NomNaNMT.txt']:
            combined_pool.append((sino, viet, 'NomNaNMT'))
        
        # Build vocabulary already covered by CLC
        covered_chars = set()
        for sino, _ in clc_lines:
            for char in sino:
                covered_chars.add(char)
        
        # Greedy selection to maximize character coverage
        n_pool = len(combined_pool)
        target_from_pool = min(remaining_needed, n_pool)
        
        # Build char_to_lines mapping for the pool
        char_to_lines = defaultdict(list)
        line_to_chars = []
        for line_idx, (sino, viet, source) in enumerate(combined_pool):
            chars_in_line = set(sino.strip())
            line_to_chars.append(chars_in_line)
            for char in chars_in_line:
                if char not in covered_chars:  # Only care about uncovered chars
                    char_to_lines[char].append(line_idx)
        
        # Initialize data structures for greedy selection
        selected = [False] * n_pool
        current_scores = []
        for i in range(n_pool):
            # Count how many new characters this line would add
            new_chars = line_to_chars[i] - covered_chars
            current_scores.append(len(new_chars))
        
        heap = []
        for i in range(n_pool):
            heapq.heappush(heap, (-current_scores[i], random.random(), i))  # Add random tiebreaker
        
        selected_indices = []
        current_covered = covered_chars.copy()
        
        print(f"Selecting {target_from_pool} lines from ThiVien and NomNaNMT to maximize coverage...")
        
        for _ in range(target_from_pool):
            # Pop valid entry from heap
            while heap:
                neg_score, _, line_idx = heapq.heappop(heap)
                score = -neg_score
                if selected[line_idx]:
                    continue
                break
            else:
                # If no more lines with new characters, randomly select remaining
                remaining_indices = [i for i in range(n_pool) if not selected[i]]
                if remaining_indices:
                    random.shuffle(remaining_indices)
                    selected_indices.extend(remaining_indices[:target_from_pool - len(selected_indices)])
                break
            
            # Select this line
            selected[line_idx] = True
            selected_indices.append(line_idx)
            
            # Update coverage
            for char in line_to_chars[line_idx]:
                if char not in current_covered:
                    current_covered.add(char)
                    # Update scores for lines containing this char
                    for other_line_idx in char_to_lines[char]:
                        if not selected[other_line_idx]:
                            current_scores[other_line_idx] -= 1
                            heapq.heappush(heap, (-current_scores[other_line_idx], random.random(), other_line_idx))
            
            # Early termination if all characters covered
            if len(current_covered) == len(overall_vocab):
                print(f"All {len(overall_vocab)} characters covered after selecting {len(selected_indices)} lines")
                break
        
        # Add selected lines to pretraining
        for idx in selected_indices:
            sino, viet, source = combined_pool[idx]
            pretraining_lines.append((sino, viet))
            
        # Randomly select remaining lines to reach 100,000 lines
        if len(pretraining_lines) < PRETRAINING_TARGET:
            look = set(pretraining_lines)
            additional_needed = PRETRAINING_TARGET - len(pretraining_lines)
            remaining_pool = [line for line in combined_pool if line not in look]
            random.shuffle(remaining_pool)
            additional_lines = remaining_pool[:additional_needed]
            for sino, viet, source in additional_lines:
                pretraining_lines.append((sino, viet))
        
        print(f"Selected {len(selected_indices)} additional lines for pretraining")
        print(f"Final coverage: {len(current_covered)}/{len(overall_vocab)} characters")
    
    # The remaining data goes to training and validation
    remaining_lines = []
    
    # Add unselected lines from all files
    selected_set = set(pretraining_lines)
    for filename, lines in file_data.items():
        for line in lines:
            if line not in selected_set:
                remaining_lines.append(line)
    
    # Shuffle remaining lines
    random.shuffle(remaining_lines)
    
    # Split into validation and training for correction
    num_val = min(args.num_val, len(remaining_lines))
    val_lines = remaining_lines[:num_val]
    correction_lines = remaining_lines[num_val:]
    
    # # Generate noisy data function
    # def generate_noisy(clean_sino):
    #     new_seq = []
    #     for i, char in enumerate(clean_sino):
    #         eligible_ops = []
            
    #         # Substitution eligibility
    #         if char in similarity_dict and similarity_dict[char]:
    #             eligible_ops.append(('sub', sub_weight))
            
    #         # Apply error if eligible, otherwise keep char
    #         if not eligible_ops:
    #             new_seq.append(char)
    #             continue
                
    #         candidates = similarity_dict[char]
    #         new_char = random.choice(candidates)
    #         new_seq.append(new_char)
                
    #     return ''.join(new_seq)

    # # Generate noisy data for validation and correction training
    # val_data = []
    # for sino, viet in val_lines:
    #     noisy_sino = generate_noisy(sino)
    #     val_data.append((noisy_sino, sino, viet))
    
    # correction_data = []
    # for sino, viet in correction_lines:
    #     noisy_sino = generate_noisy(sino)
    #     correction_data.append((noisy_sino, sino, viet))

    # # Collect all characters including noisy versions
    # all_chars = overall_vocab.copy()
    
    # for noisy_sino, _, _ in val_lines:
    #     for char in noisy_sino.strip():
    #         all_chars.add(char)
            
    # for noisy_sino, _, _ in correction_data:
    #     for char in noisy_sino.strip():
    #         all_chars.add(char)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save vocabulary
    vocab_path = os.path.join(args.output_dir, 'vocab.txt')
    with open(vocab_path, 'w', encoding='utf-8') as f_vocab:
        for char in sorted(overall_vocab):
            f_vocab.write(char + '\n')

    # Save pretraining data (clean Sino-Nom only)
    pretrain_path = os.path.join(args.output_dir, 'pretraining.txt')
    with open(pretrain_path, 'w', encoding='utf-8') as f_pretrain:
        for sino, viet in pretraining_lines:
            f_pretrain.write(sino + '\n')

    # Save correction training data (noisy-clean pairs)
    train_path = os.path.join(args.output_dir, 'train.txt')
    with open(train_path, 'w', encoding='utf-8') as f_train:
        for sino, viet in correction_lines:
            f_train.write(f"{sino}\t{viet}\n")

    # Save validation data
    val_path = os.path.join(args.output_dir, 'val.txt')
    with open(val_path, 'w', encoding='utf-8') as f_val:
        for sino, viet in val_lines:
            f_val.write(f"{sino}\t{viet}\n")

    # Print statistics
    print(f"\n✅ Successfully generated:")
    print(f"  - Vocabulary: {len(overall_vocab)} unique characters (saved to vocab.txt)")
    print(f"  - Pretraining: {len(pretraining_lines)} lines (clean Sino-Nom only)")
    print(f"  - Correction training: {len(correction_lines)} lines (noisy-clean pairs)")
    print(f"  - Validation data: {len(val_lines)} lines")
    print(f"\nOutput files:")
    print(f"  - {vocab_path}")
    print(f"  - {pretrain_path}")
    print(f"  - {train_path}")
    print(f"  - {val_path}")

if __name__ == "__main__":
    main()