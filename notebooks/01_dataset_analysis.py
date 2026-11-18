
import pandas as pd
import os

print("Starting dataset analysis...")

DATASET_PATH = os.path.join(
    "dataset",
    "PhiUSIIL_Phishing_URL_Dataset.csv"
)

print("Checking dataset file...")

if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset file not found!")
else:
    print("Dataset file found!")

    df = pd.read_csv(DATASET_PATH)

    print("\n========== DATASET SHAPE ==========")
    print(df.shape)

    print("\n========== FIRST 5 RECORDS ==========")
    print(df.head())

    print("\n========== COLUMN NAMES ==========")
    print(df.columns.tolist())

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== LAST 10 COLUMN NAMES ==========")
    print(df.columns[-10:].tolist())
    
print("\nAnalysis completed!")

print("\n========== LABEL DISTRIBUTION ==========")

print(df["label"].value_counts())

print("\n========== LABEL PERCENTAGE ==========")

print(
    df["label"].value_counts(normalize=True) * 100
)

print("\n========== DATA TYPES ==========")

print(df.dtypes.value_counts())

print("\n========== DUPLICATE ROWS ==========")

print(df.duplicated().sum())

print("\n========== SAMPLE LEGITIMATE URLS ==========")

print(
    df[df["label"] == 1][["URL", "label"]].head(5).to_string(index=False)
)

print("\n========== SAMPLE PHISHING URLS ==========")

print(
    df[df["label"] == 0][["URL", "label"]].head(5).to_string(index=False)
)

print("\n========== URL FEATURE EXAMPLES ==========")

columns = [
    "URL",
    "URLLength",
    "IsDomainIP",
    "NoOfSubDomain",
    "IsHTTPS",
    "NoOfQMarkInURL",
    "NoOfDegitsInURL",
    "label"
]

print(df[columns].sample(10, random_state=42).to_string(index=False))
