import argparse
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def parse_links(file_path):
    items = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                id_, url, status = parts
                items.append({'id': id_, 'url': url, 'status': int(status)})
    return items

def write_links(file_path, items):
    with open(file_path, 'w') as f:
        for item in items:
            f.write(f"{item['id']},{item['url']},{item['status']}\n")

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_batch(batch, function_url, headers, output_dir, session):
    urls = [item['url'] for item in batch]
    try:
        response = session.post(function_url, json={'urls': urls}, headers=headers)
        response.raise_for_status()
        results = response.json()['results']
        # Assuming results are in the same order as urls
        for i, res in enumerate(results):
            item = batch[i]
            if res.get('success', False):
                file_path = os.path.join(output_dir, f"{item['id']}.html")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(res['content'])
                item['status'] = 1
            else:
                item['status'] = 0
    except Exception as e:
        # If the entire batch fails, mark all as 0
        for item in batch:
            item['status'] = 0

def main():
    parser = argparse.ArgumentParser(description="CLI program to fetch HTML content via Supabase edge function.")
    parser.add_argument('--local', action='store_true', help="Use local Supabase instance for testing.")
    parser.add_argument('--input', required=True, help="Input links.")
    parser.add_argument('--batch', type=int, default=10, help="Batch size for URLs per request.")
    parser.add_argument('--max_workers', type=int, default=64, help="Maximum number of worker threads to use for processing batches of URLs.")
    parser.add_argument('--output-dir', required=True, help="Output directory to save HTML files.")
    args = parser.parse_args()

    if args.local:
        function_url = 'http://127.0.0.1:54321/functions/v1/http-proxy'
    else:
        supabase_url = 'https://ifjvmbrpwypgnmxcfvsl.supabase.co/functions/v1/http-proxy'
        function_url = f"{supabase_url}/functions/v1/http-proxy"

    headers = {
        'Content-Type': 'application/json'
    }

    links_file = args.input
    if not os.path.exists(links_file):
        raise FileNotFoundError(f"links.txt not found in {args.input}")

    all_items = parse_links(links_file)
    to_process = [item for item in all_items if item['status'] == 0]

    if not to_process:
        print("No URLs to process.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    batch_size = args.batch
    batches = list(chunks(to_process, batch_size))

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_batch, batch, function_url, headers, args.output_dir, session)
            for batch in batches
        ]
    
        # tqdm progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            future.result()

    # Write updated statuses back to links.txt
    write_links(links_file, all_items)

    # Track and print failed ones
    failed = [item for item in to_process if item['status'] == 0]
    if failed:
        print(f"Total failed URLs after retries: {len(failed)}")

if __name__ == '__main__':
    main()