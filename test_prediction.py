import joblib
import pandas as pd

from feature_extraction import extract_url_features


# ==============================
# Load Models
# ==============================

rf_model = joblib.load("models/random_forest_model.pkl")
svm_model = joblib.load("models/svm_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")


# ==============================
# Test URLs
# ==============================

test_urls = [
    "https://example.com",
    "https://google.com",
    "https://github.com",
    "https://microsoft.com",
    "https://wikipedia.org",
    "https://openai.com",
    "https://python.org",
    "https://stackoverflow.com"
]


# ==============================
# Test Predictions
# ==============================

print("\n========== MODEL COMPARISON ==========")

for url in test_urls:

    features = extract_url_features(url)

    df = pd.DataFrame([features])

    rf_prediction = rf_model.predict(df)[0]
    svm_prediction = svm_model.predict(df)[0]
    xgb_prediction = xgb_model.predict(df)[0]

    print("\nURL:", url)

    print(
        "Random Forest:",
        "Legitimate" if rf_prediction == 1 else "Phishing"
    )

    print(
        "SVM:",
        "Legitimate" if svm_prediction == 1 else "Phishing"
    )

    print(
        "XGBoost:",
        "Legitimate" if xgb_prediction == 1 else "Phishing"
    )