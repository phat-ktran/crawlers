import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import uuid

import pandas as pd
from bs4 import BeautifulSoup

def parse_html(file_path: Path) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Extract book information
    header = soup.find('header', class_='page-header')
    timeline = ''
    author = ''
    book_name = ''
    if header:
        breadcrum = header.find('p', class_='breadcrum')
        if breadcrum:
            a_tags = breadcrum.find_all('a')
            if len(a_tags) >= 4:
                timeline = a_tags[2].text.strip()
                author = a_tags[3].text.strip()
                book_name = a_tags[-1].text.strip()
    
    # Extract book content
    div = soup.find('div', class_='poem-view-separated')
    chinese = ''
    transcription = ''
    translation = ''
    if div:
        h4s = div.find_all('h4')
        if len(h4s) >= 3:
            # Chinese content
            p_chinese = h4s[0].find_next_sibling('p', class_='HanChinese')
            if p_chinese:
                chinese = '\n'.join(p_chinese.stripped_strings)
            
            # Transcription content
            p_transcription = h4s[1].find_next_sibling('p')
            if p_transcription:
                transcription = '\n'.join(p_transcription.stripped_strings)
            
            # Translation content
            p_translation = h4s[2].find_next_sibling('p')
            if p_translation:
                translation = '\n'.join(p_translation.stripped_strings)
    
    return {
        'ID': file_path.stem,
        'Book_name': book_name,
        'Author': author,
        'timeline': timeline,
        'Chinese': chinese,
        'Transcription': transcription,
        'Translation': translation
    }

def main():
    parser = argparse.ArgumentParser(description='Process HTML files to extract book information and content.')
    parser.add_argument('--input-dir', required=True, help='Directory containing HTML files.')
    parser.add_argument('--output-dir', required=True, help='Directory to save outputs.')
    parser.add_argument('--max_workers', type=int, default=32, help='Maximum number of workers for parallel processing.')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    xlsx_dir = output_dir / 'xlsx'
    json_dir = output_dir / 'json'
    
    os.makedirs(xlsx_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    html_files = list(input_dir.glob('*.html'))
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(executor.map(parse_html, html_files))
    
    # Create XLSX file
    df = pd.DataFrame(results)
    xlsx_path = xlsx_dir / 'output.xlsx'
    df.to_excel(xlsx_path, index=False, columns=['ID', 'Book_name', 'Author', 'timeline', 'Chinese', 'Transcription', 'Translation'])
    
    # Create JSON files
    counter = 0
    for result in results:
        book_name = result['Book_name']
        chinese = result['Chinese']
        if book_name and chinese:
            counter += 1
            json_data = {
                'book_name': result['Book_name'],
                'author': result['Author'],
                'timeline': result['timeline'],
                'chinese': result['Chinese'],
                'transcription': result['Transcription'],
                'translation': result['Translation']
            }
            json_path = json_dir / f'{uuid.uuid4()}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
                
    print(f'Total books saved as JSON: {counter}')

if __name__ == '__main__':
    main()