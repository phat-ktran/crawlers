import os
import argparse

def extract_chinese_vocab(input_dir, label_file):
    chinese_vocab = set()
    excluded_chars = set(" !(),-.0123456789:?AEIKMNO[]ahiu·đềồ​—“”…◆○●⿰⿱《》「」『』【】〔〕")
    with open(label_file, 'w', encoding='utf-8') as label_output:
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path) and filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        parts = line.strip().split('\t', 1)
                        if len(parts) == 1:
                            continue
                        chinese_seq, vietnamese = parts
                        filtered_seq = ''.join(char for char in chinese_seq if char not in excluded_chars)
                        label_output.write(f"{filtered_seq}\t{vietnamese}\n")
                        chinese_vocab.update(filtered_seq)
    return chinese_vocab

def main():
    parser = argparse.ArgumentParser(description="Extract Chinese vocabulary from text files.")
    parser.add_argument('--input-dir', required=True, help="Directory containing text files.")
    parser.add_argument('--output', required=True, help="Output file to save Chinese vocabulary.")
    parser.add_argument('--label', required=True, help="Output label file combining all text files.")
    args = parser.parse_args()

    chinese_vocab = extract_chinese_vocab(args.input_dir, args.label)
    with open(args.output, 'w', encoding='utf-8') as output_file:
        for char in sorted(chinese_vocab):
            output_file.write(f"{char}\n")

if __name__ == "__main__":
    main()
