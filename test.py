import pandas as pd

# Read the CSV file
df = pd.read_csv('data/all_with_happiness.csv')

for column in df.columns:
    print(f"{column}")