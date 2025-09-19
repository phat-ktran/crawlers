#!/usr/bin/env python3
"""
Script to generate URLs for thivien.net poetry search with all combinations
of poem types (Thể thơ) and eras (Thời đại) for Vietnam with Chinese characters.
"""

import os


def main():
    # Base URL pattern
    base_url = "https://www.thivien.net/search-poem.php?PoemType={}&ViewType=2&Country=2&Age%5B%5D={}"

    # Poem types (Thể thơ) - extracted from HTML
    poem_types = [
        (13, "Lục bát"),
        (14, "Song thất lục bát"),
        (24, "Ca trù (hát nói)"),
        (15, "Thơ mới bốn chữ"),
        (16, "Thơ mới năm chữ"),
        (17, "Thơ mới sáu chữ"),
        (18, "Thơ mới bảy chữ"),
        (19, "Thơ mới tám chữ"),
        (20, "Thơ tự do"),
        (27, "Kinh thi"),
        (10, "Tứ ngôn"),
        (5, "Ngũ ngôn cổ phong"),
        (8, "Thất ngôn cổ phong"),
        (2, "Phú"),
        (3, "Ngũ ngôn tứ tuyệt"),
        (4, "Ngũ ngôn bát cú"),
        (6, "Thất ngôn tứ tuyệt"),
        (7, "Thất ngôn bát cú"),
        (23, "Đường luật biến thể"),
        (11, "Từ phẩm"),
        (12, "Tản khúc"),
        (28, "Câu đối"),
        (30, "Tản văn"),
        (29, "Thể loại khác (thơ)"),
        (22, "Thể loại khác (ngoài thơ)"),
    ]

    # Eras (Thời đại) for Vietnam (Country=2) - extracted from JavaScript
    eras = [
        (1, "Trung đại (938 ÷ 1887)"),
        (50, "Ngô, Đinh, Tiền Lê (938 ÷ 1009)"),
        (51, "Lý-Trần (1009 ÷ 1427)"),
        (52, "Lý (1009 ÷ 1225)"),
        (53, "Trần (1225 ÷ 1400)"),
        (54, "Hồ, thuộc Minh (1400 ÷ 1427)"),
        (55, "Hậu Lê, Mạc, Trịnh-Nguyễn (1427 ÷ 1778)"),
        (56, "Tây Sơn (1778 ÷ 1802)"),
        (57, "Nguyễn (1802 ÷ 1887)"),
        (2, "Cận đại (1887 ÷ 1930)"),
    ]

    # Generate all combinations
    links = []

    for poem_id, poem_name in poem_types:
        for era_id, era_name in eras:
            url = base_url.format(poem_id, era_id)
            for page in range(1,11):
                # Create unique ID by combining poem_id and era_id
                unique_id = f"{poem_id}_{era_id}_{page}"
                links.append(f"{unique_id},{url}&Page={page},0")

    # Ensure output directory exists
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    # Write to file
    output_file = os.path.join(output_dir, "links.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for link in links:
            f.write(link + "\n")

    print(f"Generated {len(links)} URLs and saved to {output_file}")
    print("Format: <ID>,<URL>,0")
    print(
        f"Combinations: {len(poem_types)} poem types × {len(eras)} eras = {len(links)} total URLs"
    )


if __name__ == "__main__":
    main()
