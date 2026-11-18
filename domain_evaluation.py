import pandas as pd
import numpy as np

from urllib.parse import urlparse

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from feature_extraction import extract_url_features


DATASET_PATH = "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"


# ==========================================
# LOAD DATASET
# ==========================================

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(DATASET_PATH)

print("Total rows:", len(df))


# ==========================================
# EXTRACT DOMAIN
# ==========================================

def extract_domain(url):

    try:

        url = str(url)

        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        hostname = urlparse(url).hostname

        if hostname:
            return hostname.lower()

        return ""

    except Exception:
        return ""


print("\n========== EXTRACTING DOMAINS ==========")

df["domain"] = df["URL"].apply(extract_domain)

print("Unique domains:", df["domain"].nunique())


# ==========================================
# REMOVE EMPTY DOMAINS
# ==========================================

df = df[df["domain"] != ""].copy()


# ==========================================
# EXTRACT FEATURES
# ==========================================

print("\n========== EXTRACTING FEATURES ==========")

feature_rows = []

for i, url in enumerate(df["URL"]):

    feature_rows.append(
        extract_url_features(url)
    )

    if (i + 1) % 10000 == 0:
        print(f"Processed {i + 1} URLs")


X = pd.DataFrame(feature_rows)

y = df["label"]

groups = df["domain"]


print("\nFeature shape:", X.shape)
print("Number of features:", X.shape[1])


# ==========================================
# DOMAIN-AWARE SPLIT
# ==========================================

print("\n========== DOMAIN-AWARE SPLIT ==========")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)


X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "Training domains:",
    groups.iloc[train_idx].nunique()
)

print(
    "Testing domains:",
    groups.iloc[test_idx].nunique()
)


# ==========================================
# MODEL 1 - RANDOM FOREST
# ==========================================

print("\n========== RANDOM FOREST ==========")

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train, y_train)

rf_pred = random_forest.predict(X_test)


# ==========================================
# MODEL 2 - SVM
# ==========================================

print("\n========== SVM ==========")

svm_model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=42
        )
    )
])


svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)


# ==========================================
# MODEL 3 - XGBOOST
# ==========================================

print("\n========== XGBOOST ==========")

xgb_model = XGBClassifier(

    n_estimators=200,

    max_depth=6,

    learning_rate=0.1,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)


xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n----------------------------------------")
    print(name)
    print("----------------------------------------")

    print(
        f"Accuracy : {accuracy * 100:.2f} %"
    )

    print(
        f"Precision: {precision * 100:.2f} %"
    )

    print(
        f"Recall   : {recall * 100:.2f} %"
    )

    print(
        f"F1 Score : {f1 * 100:.2f} %"
    )

    print("\nConfusion Matrix:")
    print(cm)


# ==========================================
# RESULTS
# ==========================================

print("\n")
print("========================================")
print("DOMAIN-AWARE MODEL COMPARISON")
print("========================================")


evaluate_model(
    "RANDOM FOREST",
    y_test,
    rf_pred
)


evaluate_model(
    "SVM",
    y_test,
    svm_pred
)


evaluate_model(
    "XGBOOST",
    y_test,
    xgb_pred
)


# ==========================================
# FINAL COMPARISON
# ==========================================

print("\n")
print("========================================")
print("ALL DOMAIN-AWARE MODELS TESTED")
print("========================================")

print("\nRandom Forest, SVM and XGBoost")
print("were evaluated using unseen domains.")