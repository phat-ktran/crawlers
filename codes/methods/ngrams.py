import argparse
import logging
import os
import json
import jieba
import csv
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChineseSpellingChecker:
    def __init__(self):
        self.bi_gram_counts = defaultdict(int)
        self.tri_gram_counts = defaultdict(int)
        self.char_counts = defaultdict(int)
        self.confusion_sets = defaultdict(list)
        self.vn_to_nom = defaultdict(list)
        self.vocab = set()
        self.alpha = 0.1  # Smoothing parameter
        self.build_dictionary()

    def build_dictionary(self):
        """Build confusion sets based on pinyin similarity"""
        dict_file = "codes/assets/dict.csv"
        with open(dict_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if len(row) >= 2:
                    char, nom = row[0], row[1]
                    self.vn_to_nom[char].extend(nom)

    def build_confusion_sets(self, texts):
        """Build confusion sets based on pinyin similarity"""
        similarity_file = "codes/assets/similarity.csv"
        with open(similarity_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=":")
            for row in reader:
                if len(row) >= 2:
                    char, similar_chars = row[0], row[1]
                    self.confusion_sets[char].extend(list(similar_chars))
        for char in self.confusion_sets:
            if char not in self.confusion_sets[char]:
                self.confusion_sets[char].append(char)
    
    def train(self, train_file):
        """Train the model from training data"""
        logger.info("Training Chinese Spelling Checker...")
        
        # Read training data
        correct_sequences = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    correct_sequences.append(parts[1])  # Correct sequence is second column
        
        # Build confusion sets from training data
        self.build_confusion_sets(correct_sequences)
        logger.info(f"Built confusion sets for {len(self.confusion_sets)} characters")
        
        # Build n-gram counts
        for seq in correct_sequences:
            chars = list(seq)
            self.vocab.update(chars)
            
            # Count characters for unigram
            for char in chars:
                self.char_counts[char] += 1
            
            # Build bi-gram counts
            for i in range(1, len(chars)):
                bi_gram = (chars[i-1], chars[i])
                self.bi_gram_counts[bi_gram] += 1
            
            # Build tri-gram counts
            for i in range(2, len(chars)):
                tri_gram = (chars[i-2], chars[i-1], chars[i])
                self.tri_gram_counts[tri_gram] += 1
        
        logger.info(f"Trained with {len(correct_sequences)} sequences")
        logger.info(f"Vocabulary size: {len(self.vocab)}")
        logger.info(f"Bi-gram counts: {len(self.bi_gram_counts)}")
        logger.info(f"Tri-gram counts: {len(self.tri_gram_counts)}")
    
    def save_model(self, model_file):
        """Save model to file"""
        # Convert tuple keys to string representations for JSON compatibility
        bi_gram_dict = {}
        for (c1, c2), count in self.bi_gram_counts.items():
            # Use a string representation of the two characters
            key_str = f"{c1}{c2}"
            bi_gram_dict[key_str] = count
        
        tri_gram_dict = {}
        for (c1, c2, c3), count in self.tri_gram_counts.items():
            # Use a string representation of the three characters
            key_str = f"{c1}{c2}{c3}"
            tri_gram_dict[key_str] = count
        
        model = {
            'bi_gram_counts': bi_gram_dict,
            'tri_gram_counts': tri_gram_dict,
            'char_counts': dict(self.char_counts),
            'confusion_sets': dict(self.confusion_sets),
            'vocab': list(self.vocab),
            'alpha': self.alpha
        }
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(model, f, ensure_ascii=False, indent=2)
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self, model_file):
        """Load model from file"""
        with open(model_file, 'r', encoding='utf-8') as f:
            model = json.load(f)
        
        # Convert string keys back to tuples for bi-grams and tri-grams
        self.bi_gram_counts = defaultdict(int)
        for key_str, count in model['bi_gram_counts'].items():
            # Convert string back to tuple (two characters)
            if len(key_str) == 2:
                c1, c2 = key_str[0], key_str[1]
                self.bi_gram_counts[(c1, c2)] = count
        
        self.tri_gram_counts = defaultdict(int)
        for key_str, count in model['tri_gram_counts'].items():
            # Convert string back to tuple (three characters)
            if len(key_str) == 3:
                c1, c2, c3 = key_str[0], key_str[1], key_str[2]
                self.tri_gram_counts[(c1, c2, c3)] = count
        
        self.char_counts = defaultdict(int, model['char_counts'])
        self.confusion_sets = {k: v for k, v in model['confusion_sets'].items()}
        self.vocab = set(model['vocab'])
        self.alpha = model['alpha']
        
        logger.info(f"Model loaded from {model_file}")
        logger.info(f"Vocabulary size: {len(self.vocab)}")
        logger.info(f"Bi-gram counts: {len(self.bi_gram_counts)}")
        logger.info(f"Tri-gram counts: {len(self.tri_gram_counts)}")
    
    def get_bi_gram_prob(self, prev_char, curr_char):
        """Get smoothed bi-gram probability"""
        count = self.bi_gram_counts.get((prev_char, curr_char), 0)
        prev_count = self.char_counts.get(prev_char, 0)
        
        # Additive smoothing
        return (count + self.alpha) / (prev_count + self.alpha * len(self.vocab))
    
    def get_tri_gram_prob(self, prev2_char, prev_char, curr_char):
        """Get smoothed tri-gram probability"""
        count = self.tri_gram_counts.get((prev2_char, prev_char, curr_char), 0)
        prev2_count = self.bi_gram_counts.get((prev2_char, prev_char), 0)
        
        # Additive smoothing
        return (count + self.alpha) / (prev2_count + self.alpha * len(self.vocab))
    
    def calculate_sentence_score(self, sentence):
        """Calculate score for a sentence using bi-gram and tri-gram"""
        chars = list(sentence)
        score = 0.0
        
        # For the first character, use unigram probability
        if len(chars) > 0:
            first_char = chars[0]
            count = self.char_counts.get(first_char, 0)
            score += np.log((count + self.alpha) / (sum(self.char_counts.values()) + self.alpha * len(self.vocab)))
        
        # For the second character, use bi-gram
        if len(chars) > 1:
            prev_char = chars[0]
            curr_char = chars[1]
            prob = self.get_bi_gram_prob(prev_char, curr_char)
            score += np.log(prob)
        
        # For the rest, use tri-gram where possible
        for i in range(2, len(chars)):
            prev2_char = chars[i-2]
            prev_char = chars[i-1]
            curr_char = chars[i]
            
            # Try tri-gram first
            prob = self.get_tri_gram_prob(prev2_char, prev_char, curr_char)
            if prob > 0:
                score += np.log(prob)
            else:
                # Fall back to bi-gram
                prob = self.get_bi_gram_prob(prev_char, curr_char)
                if prob > 0:
                    score += np.log(prob)
                else:
                    # Fall back to unigram
                    count = self.char_counts.get(curr_char, 0)
                    prob = (count + self.alpha) / (sum(self.char_counts.values()) + self.alpha * len(self.vocab))
                    score += np.log(prob)
        
        return score
    
    def correct_sentence(self, error_sentence, vn_seq=None):
        """Correct a single sentence"""
        # Segment the sentence for context
        words = jieba.cut(error_sentence)
        word_list = list(words)
        
        # Prepare Vietnamese
        vn_words = [word.strip() for word in vn_seq.split()] if vn_seq else []
        
        # Get word boundaries for each character
        char_word_boundaries = []
        current_word = ""
        for word in word_list:
            for char in word:
                char_word_boundaries.append(len(word))
            current_word = word
        
        # Build candidate sentences
        chars = list(error_sentence)
        best_sentence = error_sentence
        best_score = float('-inf')
        
        # Try replacing each character with candidates from confusion set
        for i in range(len(chars)):
            original_char = chars[i]
            word_length = char_word_boundaries[i]
            
            # Generate candidates (skip original character)
            candidates = [c for c in self.confusion_sets.get(original_char, [original_char]) if c != original_char]
            if not candidates:
                candidates = [original_char]
            if vn_seq and self.vn_to_nom[vn_words[i]]:
                candidates.extend(self.vn_to_nom[vn_words[i]])
            
            for cand in candidates:
                # Create candidate sentence
                candidate_chars = chars.copy()
                candidate_chars[i] = cand
                candidate_sentence = ''.join(candidate_chars)
                
                # Calculate score
                score = self.calculate_sentence_score(candidate_sentence)
                
                # Update best candidate
                if score > best_score:
                    best_score = score
                    best_sentence = candidate_sentence
        
        return best_sentence
    
    def evaluate(self, val_file, include_vn):
        """Evaluate on validation data"""
        total_cer = 0.0
        count = 0
        
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                
                error_seq = parts[0]
                correct_seq = parts[1]
                vn_seq = parts[-1] if include_vn else None
                
                # Correct the sentence
                corrected = self.correct_sentence(error_seq, vn_seq)
                
                # Calculate CER (Character Error Rate)
                dist = self.levenshtein_distance(corrected, correct_seq)
                cer = dist / len(correct_seq) if len(correct_seq) > 0 else 0
                total_cer += cer
                count += 1
                
                logger.info(f"Evaluated: Error='{error_seq}', Corrected='{corrected}', Correct='{correct_seq}', CER={cer:.4f}")
        
        avg_cer = total_cer / count if count > 0 else 0
        logger.info(f"Average CER: {avg_cer:.4f}")
        return avg_cer
    
    def levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

def main():
    parser = argparse.ArgumentParser(description="Chinese Spelling Check System based on N-gram Model")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing train.txt and val.txt")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save model files")
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode")
    parser.add_argument("--include_vn", action="store_true", help="Include Vietnamese context")
    parser.add_argument("--model", type=str, help="Path to model file for evaluation")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize checker
    checker = ChineseSpellingChecker()
    
    if args.eval:
        # Load model
        model_path = args.model if args.model else os.path.join(args.output_dir, "model.json")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return
        
        checker.load_model(model_path)
        
        # Evaluate
        val_file = os.path.join(args.input_dir, "val.txt")
        if not os.path.exists(val_file):
            logger.error(f"Validation file not found at {val_file}")
            return
        
        logger.info("Starting evaluation...")
        avg_cer = checker.evaluate(val_file, args.include_vn)
        logger.info(f"Final evaluation result - Average CER: {avg_cer:.4f}")
    else:
        # Train model
        train_file = os.path.join(args.input_dir, "train.txt")
        if not os.path.exists(train_file):
            logger.error(f"Training file not found at {train_file}")
            return
        
        logger.info("Starting training...")
        checker.train(train_file)
        
        # Save model
        model_path = os.path.join(args.output_dir, "model.json")
        checker.save_model(model_path)
        logger.info("Training completed successfully")

if __name__ == "__main__":
    main()