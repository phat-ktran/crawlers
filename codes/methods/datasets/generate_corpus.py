import random

# Load and sample 100000 lines from ThiVien.txt
with open('data/training/normalized/ThiVien.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
sampled_lines = random.sample(lines, min(100000, len(lines)))

# Extract the first element from each sampled line
processed_lines = [line.split('\t')[0] for line in sampled_lines]

# Load lines from CLC.txt
with open('data/training/normalized/CLC.txt', 'r', encoding='utf-8') as file:
    clc_lines = file.readlines()

# Combine the processed lines with CLC lines to form a new text corpus
new_corpus = processed_lines + clc_lines

# Save the new text corpus to a target file
with open('data/training/corpus/pretraining.txt', 'w', encoding='utf-8') as target_file:
    target_file.writelines(new_corpus)
    
