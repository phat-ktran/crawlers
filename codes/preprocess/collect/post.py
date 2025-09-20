import argparse
import os
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def parse_corpus(file_path):
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            text = line.strip()
            if text:
                items.append({'id': i, 'text': text, 'status': 0})
    return items

def write_results(file_path, results):
    with open(file_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"{result['hannom_text']}\t{result['transcription']}\n")

def run_curl(item):
    text = item['text'].replace("\n", "\\n")

    curl_cmd = [
        "curl", "--insecure",
        "https://kimhannom.clc.hcmus.edu.vn/api/web/clc-sinonom/sinonom-transliteration",
        "-X", "POST",
        "-H", "User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0",
        "-H", "Accept: */*",
        "-H", "Accept-Language: en-US,en;q=0.5",
        "-H", "Accept-Encoding: gzip, deflate, br, zstd",
        "-H", "Referer: https://kimhannom.clc.hcmus.edu.vn/",
        "-H", "Content-Type: application/json",
        "-H", "Origin: https://kimhannom.clc.hcmus.edu.vn",
        "-H", "Connection: keep-alive",
        "-H", "Cookie: _ga_V1TN16GB6T=GS2.1.s1751173510$o9$g0$t1751173518$j52$l0$h0; _ga=GA1.3.1861809887.1738672956; _clck=17qd92k%7C2%7Cfx6%7C0%7C1861; _ga_FV58V7FLLC=GS2.3.s1757956543$o29$g0$t1757956543$j60$l0$h0; _cc_id=3c267b1a79ff7a6703777798fbc6a657; _ga_97T4F75GB6=GS2.1.s1757829473$o16$g1$t1757829939$j49$l0$h0; _ga_SXJZTF7197=GS2.1.s1754579723$o3$g1$t1754579734$j49$l0$h0; _ga_GGNB38CBCZ=GS2.1.s1754988436$o1$g0$t1754988438$j58$l0$h0",
        "-H", "Sec-Fetch-Dest: empty",
        "-H", "Sec-Fetch-Mode: cors",
        "-H", "Sec-Fetch-Site: same-origin",
        "-H", "Priority: u=4",
        "--data-raw", json.dumps({"text": text, "lang_type": 0, "font_type": 1})
    ]

    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        print(data)

        if data.get("is_success", False):
            hannom_text = data["data"]["result_hannom_text"][0]
            transcription = data["data"]["result_text_transcription"][0]
            item["hannom_text"] = hannom_text
            item["transcription"] = transcription
            item["status"] = 1
        else:
            item["status"] = 0

    except Exception as e:
        item["status"] = 0
    return item

def main():
    parser = argparse.ArgumentParser(description="CLI program to process text corpus via curl subprocess calls.")
    parser.add_argument('--input', required=True, help="Input text corpus file.")
    parser.add_argument('--output', required=True, help="Output file to save results.")
    parser.add_argument('--max_workers', type=int, default=8, help="Maximum number of worker threads.")
    args = parser.parse_args()

    corpus_file = args.input
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"Input file not found at {args.input}")

    all_items = parse_corpus(corpus_file)
    to_process = [item for item in all_items if item['status'] == 0]

    if not to_process:
        print("No texts to process.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_curl, item): item for item in to_process}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())

    successful = [item for item in results if item['status'] == 1]
    if successful:
        write_results(args.output, successful)
        print(f"Saved {len(successful)} results to {args.output}")

    failed = [item for item in results if item['status'] == 0]
    if failed:
        print(f"Total failed: {len(failed)}")
    

if __name__ == '__main__':
    main()
