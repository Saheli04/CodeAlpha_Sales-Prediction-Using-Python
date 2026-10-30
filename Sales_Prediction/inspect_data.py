
import pandas as pd

# Load the dataset
df = pd.read_csv("Advertising.csv")

print("SALES PREDICTION - DATA INSPECTION")
print("=" * 40)

print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())

# Remove the index-like column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("\nCleaned dataset shape:", df.shape)
print("\nData inspection completed successfully!")
