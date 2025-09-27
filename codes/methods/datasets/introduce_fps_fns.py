import argparse
import random
from collections import defaultdict
import csv

def main():
    parser = argparse.ArgumentParser(description="Augment a dataset for Sino-Nom correction by introducing false positives and false negatives in approximated binary sequences.")
    parser.add_argument('--input', required=True, help="Input text file with lines in the format: <input Sino-Nom char seq>\t<target Sino-Nom char seq>\t<approx binary seq sep by comma>\t<Vietnamese word seq>.")
    parser.add_argument('--output', required=True, help="Output text file with augmented content in the same format.")
    parser.add_argument('--vocab', required=True, help="Text file containing the Sino-Nom vocabulary, with each line as a character.")
    parser.add_argument('--dict', required=True, help="CSV dictionary file for Sino-Nom to Vietnamese, with columns 'QuocNgu' and 'SinoNom'.")
    parser.add_argument('--similarity', required=True)
    args = parser.parse_args()

    # Load vocabulary as a set for quick lookup
    with open(args.vocab, 'r', encoding='utf-8') as f:
        vocab = {line.strip() for line in f if line.strip()}

    # Load dictionary as a mapping from Vietnamese words to sets of Sino-Nom characters
    dict_viet_to_sino = defaultdict(set)
    with open(args.dict, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            quoc = row['QuocNgu'].strip()
            sino = row['SinoNom'].strip()
            if quoc and sino:
                dict_viet_to_sino[quoc].add(sino)
                
    # Load dictionary as a mapping from Vietnamese words to sets of Sino-Nom characters
    similarity = defaultdict(set)
    with open(args.similarity, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=":")
        for row in reader:
            sino = row['Char'].strip()
            candidates = [char for char in list(row['Candidates'].strip()) if char in vocab]
            candidates.append(sino)
            if sino in vocab:
                similarity[sino].update(candidates)

    # Process input and write to output
    with open(args.output, 'w', encoding='utf-8') as out_f:
        with open(args.input, 'r', encoding='utf-8') as in_f:
            counter = 0
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 4:
                    continue
                X_str = parts[0].strip()
                Y_str = parts[1].strip()
                B_prime_str = parts[2].strip()
                T_str = parts[3].strip()

                N = len(X_str)
                if N != len(Y_str) or N == 0:
                    continue

                X = list(X_str)
                Y = list(Y_str)
                T = T_str.split()
                if len(T) != N:
                    continue

                try:
                    B_prime = [int(b) for b in B_prime_str.split(',')]
                    if len(B_prime) != N:
                        continue
                except ValueError:
                    continue

                # Compute ground-truth binary sequence B
                true_B = [1 if X[i] == Y[i] else 0 for i in range(N)]

                # Skip augmentation for short sequences
                if N < 8:
                    out_f.write(line + '\n')
                    continue

                # Create augmented version
                new_X = X.copy()

                # Introduce false positives (20-30% of characters)
                fp_perc = random.uniform(0.3, 0.5)
                num_fp = int(fp_perc * N)
                candidates_fp = [i for i in range(N) if true_B[i] == 1 and len([z for z in dict_viet_to_sino[T[i]] if z != new_X[i] and z in vocab]) > 0]
                random.shuffle(candidates_fp)
                selected_fp = candidates_fp[:num_fp]
                
                for i in selected_fp:
                    possible_Z = [z for z in dict_viet_to_sino[T[i]] if z != new_X[i] and z in similarity[new_X[i]]]
                    if possible_Z:
                        counter += 1
                        Z = random.choice(possible_Z)
                        new_X[i] = Z
                        continue
                    possible_Z = [z for z in dict_viet_to_sino[T[i]] if z != new_X[i] and z in vocab]
                    if possible_Z:
                        Z = random.choice(possible_Z)
                        new_X[i] = Z
                
                # Recompute ground-truth B for the new X
                new_true_B = [1 if new_X[i] == Y[i] else 0 for i in range(N)]

                # Compute dictionary-based B' for the new X
                B_dict = [1 if new_X[i] in dict_viet_to_sino[T[i]] else 0 for i in range(N)]

                # Introduce false negatives (10% of characters)
                num_fn = int(0.1 * N)
                candidates_fn = [i for i in range(N) if new_true_B[i] == 1 and B_dict[i] == 1]
                random.shuffle(candidates_fn)
                selected_fn = candidates_fn[:num_fn]
                for i in selected_fn:
                    B_dict[i] = 0  # Artificially flip to simulate false negative

                # Construct and write the augmented line
                new_X_str = ''.join(new_X)
                new_B_prime_str = ','.join(map(str, B_dict))
                new_line = '\t'.join([new_X_str, Y_str, new_B_prime_str, T_str])
                out_f.write(new_line + '\n')
                
            print(counter)

if __name__ == '__main__':
    main()