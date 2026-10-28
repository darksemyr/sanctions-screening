import pandas as pd

# Load dataset
df = pd.read_parquet("data/interim/pairs_flat.parquet")

# Dataset dimensions
print("SHAPE:")
print(df.shape)

# Column names
print("\nCOLUMNS:")
print(df.columns.tolist())

# First 5 rows
print("\nFIRST 5 ROWS:")
print(df.head())

# Data types
print("\nDATA TYPES:")
print(df.dtypes)