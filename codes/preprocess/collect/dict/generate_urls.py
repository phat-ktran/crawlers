import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('data/soict/dict.csv')

# Extract unique values from the 'ChuNom' column
unique_values = df['SinoNom'].dropna().unique()

# Save the unique values to a text file, line by line
with open('output.txt', 'w', encoding='utf-8') as f:
    for value in unique_values:
        f.write(f"{value},https://hvdic.thivien.net/whv/{value},0\n")