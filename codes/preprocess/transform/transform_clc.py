import argparse
import pandas as pd
import re


def normalize_first_column(value):
    # Remove "。", "，" from the first column
    normalized = (
        value.replace("。", "")
        .replace("，", "")
        .replace("？", "")
        .replace("、", "")
        .replace("：", "")
        .replace("！", "")
        .replace("；", "")
        .replace("－", "")
        .replace("（", "")
        .replace("）", "")
        .strip()
    )

    return normalized


def normalize_second_column(value):
    # Convert to lowercase and remove special characters
    normalized =  re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", value).lower().strip())

    # Replace numbers 1-9 with their Vietnamese equivalents
    vietnamese_numbers = {
        "1": "một",
        "2": "hai",
        "3": "ba",
        "4": "bốn",
        "5": "năm",
        "6": "sáu",
        "7": "bảy",
        "8": "tám",
        "9": "chín",
    }
    for num, vietnamese in vietnamese_numbers.items():
        normalized = normalized.replace(num, vietnamese)

    return normalized

def main():
    parser = argparse.ArgumentParser(
        description="Process an xlsx file and extract specific columns."
    )
    parser.add_argument("--input", required=True, help="Path to the input xlsx file.")
    parser.add_argument("--output", required=True, help="Path to the output text file.")
    args = parser.parse_args()

    # Read the input xlsx file
    df = pd.read_excel(args.input)

    # Ensure the required columns exist
    required_columns = ["Chữ Hán", "Âm Hán Việt", "Nghĩa thuần Việt"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Normalize and extract the first two columns
    lines = []
    for _, row in df.iterrows():
        first_col = normalize_first_column(str(row["Chữ Hán"]))
        second_col = normalize_second_column(str(row["Âm Hán Việt"]))
        lines.append(f"{first_col}\t{second_col}")

    # Write the output to the specified file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
