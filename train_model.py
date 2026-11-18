import pandas as pd
import os
import joblib

from feature_extraction import extract_url_features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# ==============================
# 1. File Paths
# ==============================

DATASET_PATH = os.path.join(
    "dataset",
    "PhiUSIIL_Phishing_URL_Dataset.csv"
)

PROCESSED_PATH = os.path.join(
    "dataset",
    "processed_features.csv"
)


# ==============================
# 2. Load / Create Features
# ==============================

if os.path.exists(PROCESSED_PATH):

    print("Loading processed features...")

    processed_df = pd.read_csv(PROCESSED_PATH)

    X = processed_df.drop("label", axis=1)
    y = processed_df["label"]

    print("Processed features loaded successfully!")

else:

    print("Processed features not found.")
    print("Loading original dataset...")

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully!")
    print("Total rows:", len(df))

    print("\nExtracting URL features...")

    feature_rows = []

    for i, url in enumerate(df["URL"]):

        features = extract_url_features(url)
        feature_rows.append(features)

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} URLs")

    X = pd.DataFrame(feature_rows)

    y = df["label"]

    # Combine features + label
    processed_df = X.copy()
    processed_df["label"] = y.values

    # Save processed features
    processed_df.to_csv(
        PROCESSED_PATH,
        index=False
    )

    print("\nProcessed features saved successfully!")
    print("Saved at:", PROCESSED_PATH)


# ==============================
# 3. Dataset Information
# ==============================

print("\n========== FEATURE DATASET ==========")

print("Feature shape:", X.shape)

print("Number of features:", X.shape[1])

print("\nLabel distribution:")
print(y.value_counts())


# ==============================
# 4. Train-Test Split
# ==============================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n========== DATASET SPLIT ==========")

print("Total samples:", len(X))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# 5. Random Forest
# ==============================

print("\n========== RANDOM FOREST TRAINING ==========")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("Random Forest training completed!")

rf_path = os.path.join(
    "models",
    "random_forest_model.pkl"
)

joblib.dump(rf_model, rf_path)

print("Random Forest model saved!")


# Evaluation
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results:")

print(
    "Accuracy :",
    round(accuracy_score(y_test, rf_pred) * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision_score(y_test, rf_pred) * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(recall_score(y_test, rf_pred) * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(f1_score(y_test, rf_pred) * 100, 2),
    "%"
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# ==============================
# 6. SVM
# ==============================

print("\n========== SVM TRAINING ==========")

svm_model = Pipeline([
    ("scaler", StandardScaler()),

    ("svm", LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42
    ))
])

svm_model.fit(X_train, y_train)

print("SVM training completed!")

svm_path = os.path.join(
    "models",
    "svm_model.pkl"
)

joblib.dump(svm_model, svm_path)

print("SVM model saved!")


# Evaluation
svm_pred = svm_model.predict(X_test)

print("\nSVM Results:")

print(
    "Accuracy :",
    round(accuracy_score(y_test, svm_pred) * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision_score(y_test, svm_pred) * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(recall_score(y_test, svm_pred) * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(f1_score(y_test, svm_pred) * 100, 2),
    "%"
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svm_pred))


# ==============================
# 7. XGBoost
# ==============================

print("\n========== XGBOOST TRAINING ==========")

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

print("XGBoost training completed!")

xgb_path = os.path.join(
    "models",
    "xgboost_model.pkl"
)

joblib.dump(xgb_model, xgb_path)

print("XGBoost model saved!")


# Evaluation
xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost Results:")

print(
    "Accuracy :",
    round(accuracy_score(y_test, xgb_pred) * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision_score(y_test, xgb_pred) * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(recall_score(y_test, xgb_pred) * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(f1_score(y_test, xgb_pred) * 100, 2),
    "%"
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))


# ==============================
# 8. Training Complete
# ==============================

print("\n========================================")
print("ALL MODELS TRAINED SUCCESSFULLY!")
print("========================================")

print("\nSaved models:")

print("1. models/random_forest_model.pkl")
print("2. models/svm_model.pkl")
print("3. models/xgboost_model.pkl")

print("\nProcessed dataset:")

print("dataset/processed_features.csv")