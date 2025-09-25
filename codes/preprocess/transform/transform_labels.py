import argparse
import py_vncorenlp
import os



def process_file(input_path: str, output_path: str, save_dir: str = "/tmp/"):
    # Ensure VnCoreNLP model is available
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        py_vncorenlp.download_model(save_dir=save_dir)
        rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
        for line in fin:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"Invalid line format (expected 4 tab-separated fields): {line}")

            sino1, sino2, binary_seq, viet_text = parts

            # Word segmentation
            segmented = rdrsegmenter.word_segment(viet_text)
            # Join tokens back into space-separated string
            viet_transformed = " ".join(segmented)

            # Write back with transformed Vietnamese sequence
            fout.write("\t".join([sino1, sino2, binary_seq, viet_transformed]) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Transform Vietnamese sequences with VnCoreNLP word segmentation.")
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--output", required=True, help="Path to output file")
    parser.add_argument("--save-dir", default="/tmp/", help="Directory to download/load VnCoreNLP models")
    args = parser.parse_args()

    process_file(args.input, args.output, args.save_dir)


if __name__ == "__main__":
    main()
