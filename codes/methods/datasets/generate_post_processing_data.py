#!/usr/bin/env python3
"""
Sino-Nom Post-OCR Processing Dataset Synthesizer

This program synthesizes training data for post-OCR correction in Sino-Nom texts
by maximizing Character Error Rate (CER) while maintaining semantic consistency.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import editdistance


class SinoNomVietDictionary:
    """Dictionary for Sino-Nom to Vietnamese mappings"""
    
    def __init__(self):
        self.sino_to_viet: Dict[str, Set[str]] = defaultdict(set)
        self.viet_to_sino: Dict[str, Set[str]] = defaultdict(set)
        
    def add_mapping(self, sino_char: str, viet_word: str):
        """Add a mapping between Sino-Nom character and Vietnamese word"""
        self.sino_to_viet[sino_char].add(viet_word)
        self.viet_to_sino[viet_word].add(sino_char)
        
    def get_vietnamese_words(self, sino_char: str) -> Set[str]:
        """Get Vietnamese words for a Sino-Nom character"""
        return self.sino_to_viet[sino_char]
        
    def get_sino_chars(self, viet_word: str) -> Set[str]:
        """Get Sino-Nom characters for a Vietnamese word"""
        return self.viet_to_sino[viet_word]
        
    def are_semantically_similar(self, sino_char1: str, sino_char2: str) -> bool:
        """Check if two Sino-Nom characters have overlapping Vietnamese meanings"""
        meanings1 = self.sino_to_viet[sino_char1]
        meanings2 = self.sino_to_viet[sino_char2]
        return len(meanings1.intersection(meanings2)) > 0


def load_sino_nom_viet_dictionary() -> SinoNomVietDictionary:
    """Load Sino-Nom Vietnamese dictionary
    
    This is a placeholder implementation. Replace with your actual dictionary loading code.
    """
    dictionary = SinoNomVietDictionary()
    
    # Read a CSV file with columns QuocNgu and SinoNom
    csv_file = 'codes/assets/dict.csv'

    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"Dictionary file not found: {csv_file}")

    sample_mappings = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) == 2:
                    quoc_ngu, sino_nom = parts
                    sample_mappings.append((sino_nom.strip('"'), quoc_ngu.strip('"')))
    
    for sino, viet in sample_mappings:
        dictionary.add_mapping(sino, viet)
        
    return dictionary


def extract_filename(path: str) -> str:
    """Extract filename from full path"""
    return os.path.basename(path)


def parse_ground_truth(gt_file: str) -> Dict[str, str]:
    """Parse ground truth file"""
    gt_mapping = {}
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    filename = parts[0]
                    sino_text = parts[1]
                    gt_mapping[filename] = sino_text
                    
    return gt_mapping


def parse_translation(translation_file: str) -> Dict[str, str]:
    """Parse translation file"""
    translation_mapping = {}
    
    with open(translation_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    sino_text = parts[0]
                    viet_text = parts[1]
                    translation_mapping[sino_text] = viet_text
                    
    return translation_mapping


def parse_ocr_result(ocr_file: str) -> Dict[str, Tuple[str, float]]:
    """Parse OCR result file"""
    ocr_mapping = {}
    
    with open(ocr_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 3:
                    path = parts[0]
                    sino_text = parts[1]
                    confidence = float(parts[2]) if parts[2] else 0.0
                    
                    filename = extract_filename(path)
                    ocr_mapping[filename] = (sino_text, confidence)
                    
    return ocr_mapping


def align_sequences(prediction: str, ground_truth: str) -> List[Tuple[Optional[str], str]]:
    """
    Align prediction to ground truth using edit distance
    Returns list of (predicted_char, gt_char) pairs
    """
    if len(prediction) == len(ground_truth):
        return list(zip(prediction, ground_truth))
    
    # Use dynamic programming for alignment
    pred_len, gt_len = len(prediction), len(ground_truth)
    
    # DP table for edit distance with traceback
    dp = [[0] * (gt_len + 1) for _ in range(pred_len + 1)]
    
    # Initialize base cases
    for i in range(pred_len + 1):
        dp[i][0] = i
    for j in range(gt_len + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, pred_len + 1):
        for j in range(1, gt_len + 1):
            if prediction[i-1] == ground_truth[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    # Traceback to get alignment
    alignment = []
    i, j = pred_len, gt_len
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if prediction[i-1] == ground_truth[j-1] else 1):
            # Match or substitution
            alignment.append((prediction[i-1], ground_truth[j-1]))
            i, j = i-1, j-1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # Deletion (char in prediction but not in ground truth)
            i -= 1
        else:
            # Insertion (char in ground truth but not in prediction)
            alignment.append((None, ground_truth[j-1]))
            j -= 1
    
    alignment.reverse()
    return alignment


def select_best_error_candidate(
    gt_char: str,
    candidates: List[str],
    gt_viet_word: str,
    dictionary: SinoNomVietDictionary
) -> str:
    """
    Select the best error candidate based on semantic similarity
    """
    if not candidates:
        return gt_char
    
    # Priority 1: Candidates with same Vietnamese meaning
    semantic_candidates = []
    for candidate in candidates:
        if dictionary.are_semantically_similar(gt_char, candidate):
            semantic_candidates.append(candidate)
    
    if semantic_candidates:
        return random.choice(semantic_candidates)
    
    # Priority 2: Random selection from all candidates
    return random.choice(candidates)

def align_sino_nom_vietnamese(
    sino_seq: str, 
    viet_words: List[str], 
    dictionary: SinoNomVietDictionary
) -> List[int]:
    """
    Align Sino-Nom sequence with Vietnamese word sequence using dictionary
    Returns binary sequence indicating accurate translation (1) or not (0)
    
    Args:
        sino_seq: Sino-Nom character sequence
        viet_words: Vietnamese word sequence
        dictionary: Sino-Nom Vietnamese dictionary
        
    Returns:
        List of binary values (0 or 1) with length equal to len(sino_seq)
    """
    if len(sino_seq) != len(viet_words):
        print(f"Warning: Length mismatch in alignment: {len(sino_seq)} chars vs {len(viet_words)} words")
        # Pad or truncate to match sino_seq length
        if len(viet_words) < len(sino_seq):
            viet_words = viet_words + [""] * (len(sino_seq) - len(viet_words))
        else:
            viet_words = viet_words[:len(sino_seq)]
    
    binary_sequence = []
    
    for i, sino_char in enumerate(sino_seq):
        viet_word = viet_words[i] if i < len(viet_words) else ""
        
        # Check if the Sino-Nom character has this Vietnamese word as a valid translation
        possible_viet_words = dictionary.get_vietnamese_words(sino_char)
        
        if viet_word and viet_word in possible_viet_words:
            binary_sequence.append(1)  # Accurate translation
        else:
            binary_sequence.append(0)  # Inaccurate translation
    
    return binary_sequence

def calculate_cer(prediction: str, ground_truth: str) -> float:
    """Calculate Character Error Rate"""
    if not ground_truth:
        return 0.0
    
    edit_distance = editdistance.eval(prediction, ground_truth)
    return edit_distance / len(ground_truth)


def synthesize_dataset(input_dir: str, output_dir: str):
    """Main function to synthesize the dataset"""
    
    # Load dictionary
    print("Loading Sino-Nom Vietnamese dictionary...")
    dictionary = load_sino_nom_viet_dictionary()
    
    # Parse input files
    print("Parsing input files...")
    gt_file = os.path.join(input_dir, 'gt.txt')
    translation_file = os.path.join(input_dir, 'translation.txt')
    
    ground_truth = parse_ground_truth(gt_file)
    translations = parse_translation(translation_file)
    
    print(f"Loaded {len(ground_truth)} ground truth entries")
    print(f"Loaded {len(translations)} translation entries")
    
    # Parse all OCR result files
    ocr_results = {}
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt') and filename not in ['gt.txt', 'translation.txt']:
            ocr_file = os.path.join(input_dir, filename)
            ocr_data = parse_ocr_result(ocr_file)
            ocr_results[filename] = ocr_data
            print(f"Loaded OCR results from {filename}: {len(ocr_data)} entries")
    
    # Synthesize training data
    print("Synthesizing training data...")
    training_data = []
    cer_scores = []
    
    for filename, gt_sino in ground_truth.items():
        if gt_sino not in translations:
            print(f"Warning: No translation found for ground truth: {gt_sino}")
            continue
            
        gt_viet = translations[gt_sino]
        gt_viet_words = gt_viet.split()
        
        # Validate condition: number of Vietnamese words equals Sino-Nom characters
        if len(gt_viet_words) != len(gt_sino):
            print(f"Warning: Length mismatch for {filename}: {len(gt_sino)} chars vs {len(gt_viet_words)} words")
            continue
        
        # Collect all predictions for this filename from different OCR models
        predictions = []
        for ocr_model_name, ocr_data in ocr_results.items():
            if filename in ocr_data:
                pred_sino, confidence = ocr_data[filename]
                predictions.append((pred_sino, confidence, ocr_model_name))
        
        if not predictions:
            # No predictions available, use ground truth (no error)
            error_sino = gt_sino
            cer = 0.0
        else:
            # Find the prediction that maximizes CER
            best_prediction = None
            max_cer = -1.0
            
            for pred_sino, confidence, model_name in predictions:
                cer = calculate_cer(pred_sino, gt_sino)
                if cer > max_cer:
                    max_cer = cer
                    best_prediction = (pred_sino, confidence, model_name)
            
            if best_prediction is None:
                error_sino = gt_sino
                cer = 0.0
            else:
                pred_sino, _, _ = best_prediction
                
                # Align prediction with ground truth
                alignment = align_sequences(pred_sino, gt_sino)
                
                # Construct error sequence with semantic-aware selection
                error_chars = []
                for i, (pred_char, gt_char) in enumerate(alignment):
                    if pred_char is None:
                        # Insertion in ground truth - use ground truth char
                        error_chars.append(gt_char)
                    elif pred_char == gt_char:
                        # Correct prediction
                        error_chars.append(gt_char)
                    else:
                        # Error case - choose best candidate
                        gt_viet_word = gt_viet_words[len(error_chars)] if len(error_chars) < len(gt_viet_words) else ""
                        selected_char = select_best_error_candidate(
                            gt_char, [pred_char], gt_viet_word, dictionary
                        )
                        error_chars.append(selected_char)
                
                error_sino = ''.join(error_chars)
                cer = calculate_cer(error_sino, gt_sino)
        
        # Add to training data with binary sequence
        binary_seq = align_sino_nom_vietnamese(error_sino, gt_viet_words, dictionary)
        binary_str = ','.join(map(str, binary_seq))
        
        training_data.append((error_sino, gt_sino, binary_str, gt_viet))
        cer_scores.append(cer)
    
    # Calculate statistics
    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0.0
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    train_file = os.path.join(output_dir, 'train.txt')
    with open(train_file, 'w', encoding='utf-8') as f:
        for error_sino, gt_sino, binary_str, viet_text in training_data:
            f.write(f"{error_sino}\t{gt_sino}\t{binary_str}\t{viet_text}\n")
    
    # Save summary statistics
    summary = {
        'total_samples': len(training_data),
        'average_cer': avg_cer,
        'cer_distribution': {
            'min': min(cer_scores) if cer_scores else 0.0,
            'max': max(cer_scores) if cer_scores else 0.0,
            'mean': avg_cer
        },
        'ocr_models_used': list(ocr_results.keys())
    }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset synthesis completed!")
    print(f"Generated {len(training_data)} training samples")
    print(f"Average CER: {avg_cer:.4f}")
    print(f"Results saved to {output_dir}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Synthesize post-OCR processing dataset for Sino-Nom texts"
    )
    
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Input directory containing gt.txt, translation.txt, and OCR result files'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for train.txt and summary.json'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    required_files = ['gt.txt', 'translation.txt']
    for filename in required_files:
        filepath = os.path.join(args.input_dir, filename)
        if not os.path.isfile(filepath):
            print(f"Error: Required file not found: {filepath}")
            return 1
    
    try:
        synthesize_dataset(args.input_dir, args.output_dir)
        return 0
    except Exception as e:
        print(f"Error during dataset synthesis: {e}")
        return 1


if __name__ == '__main__':
    exit(main())