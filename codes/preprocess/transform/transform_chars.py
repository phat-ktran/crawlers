import csv
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process a CSV file.")
parser.add_argument("input_csv", help="Path to the input CSV file")
parser.add_argument("output_csv", help="Path to the output CSV file")
args = parser.parse_args()

# Load the CSV file
with open(args.input_csv, mode='r', encoding='utf-8') as infile:
    reader = list(csv.DictReader(infile))
    rows = [row for row in reader]

# Track the number of rows updated
updated_rows_count = 0

# Process rows where Strokes is empty but Radicals is not empty
for row in rows:
    if row['Strokes'] == '' and row['Radicals'] != '':
        radicals = row['Radicals']
        strokes_values = []

        # For each character in Radicals, find corresponding Strokes value
        for char in radicals:
            for original_row in rows:
                if original_row['ID'] == char:
                    strokes_values.append(original_row['Strokes'])
                    break

        # Update the Strokes column with the collected values
        row['Strokes'] = ''.join(strokes_values)
        updated_rows_count += 1

# Log the number of rows updated
logging.info(f"Number of rows updated: {updated_rows_count}")

# Save the updated CSV to the output path
with open(args.output_csv, mode='w', encoding='utf-8', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=reader[0].keys())
    writer.writeheader()
    writer.writerows(rows)
