import pandas as pd
from urllib.parse import urlparse

DATASET_PATH = "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("\n========== DATASET CHECK ==========")

print("Total rows:", len(df))

print("\nFirst 10 URLs:")

print(
    df[["URL", "label"]].head(10).to_string(index=False)
)

print("\nRandom 20 URLs:")

print(
    df[["URL", "label"]]
    .sample(20, random_state=42)
    .to_string(index=False)
)

print("\nLabel counts:")

print(df["label"].value_counts())

print("\nUnique URLs:", df["URL"].nunique())

print(
    "\nDuplicate URLs:",
    df["URL"].duplicated().sum()
)

# ==========================================================
# LEGITIMATE URL ANALYSIS
# ==========================================================

print("\n========== LEGITIMATE URL ANALYSIS ==========")

legitimate = df[df["label"] == 1].copy()

print("Total legitimate URLs:", len(legitimate))

print("\nSample legitimate URLs:")

print(
    legitimate[["URL", "label"]]
    .sample(30, random_state=42)
    .to_string(index=False)
)

# Extract domains
legitimate["domain"] = legitimate["URL"].astype(str).apply(
    lambda x: urlparse(x).netloc.lower()
)

legitimate["domain"] = legitimate["domain"].str.replace(
    "^www\\.", "", regex=True
)

print("\nTop 30 legitimate domains:")

print(
    legitimate["domain"]
    .value_counts()
    .head(30)
    .to_string()
)

# Check famous domains
test_domains = [
    "example.com",
    "google.com",
    "github.com",
    "microsoft.com",
    "wikipedia.org",
    "openai.com",
    "python.org",
    "stackoverflow.com"
]

print("\n========== FAMOUS DOMAIN CHECK ==========")

for domain in test_domains:

    matches = legitimate[
        legitimate["domain"] == domain
    ]

    print(
        f"{domain}: {len(matches)} legitimate URLs"
    )

    if len(matches) > 0:
        print(
            matches["URL"].head(3).to_string(index=False)
        )
        print("\n========== DOMAIN FREQUENCY ANALYSIS ==========")

domain_counts = legitimate["domain"].value_counts()

print("Total unique legitimate domains:", len(domain_counts))

print("\nDomains appearing more than once:")
print(
    (domain_counts > 1).sum()
)

print("\nDomains appearing 5+ times:")
print(
    (domain_counts >= 5).sum()
)

print("\nDomains appearing 10+ times:")
print(
    (domain_counts >= 10).sum()
)

print("\nTop 20 legitimate domains by URL count:")

print(
    domain_counts.head(20).to_string()
)