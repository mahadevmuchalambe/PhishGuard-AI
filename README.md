# 🛡️ PhishGuard AI — Phishing Website Detection Using Machine Learning

PhishGuard AI is a machine learning-based web application that analyzes website URLs and predicts whether they are potentially phishing or legitimate.

The system extracts 59 URL-based numerical features and uses three machine learning classification algorithms:

- Random Forest
- Support Vector Machine (SVM)
- XGBoost

The project provides a Flask-based web interface where users can enter a URL and view predictions from all three models.

## 🚀 Project Overview

Phishing websites are designed to look legitimate while attempting to trick users into providing sensitive information.

PhishGuard AI provides an automated way to analyze suspicious URLs using machine learning.

### Workflow

```text
User enters URL
       ↓
URL Feature Extraction
       ↓
59 Numerical Features
       ↓
┌─────────────────────────────┐
│ Random Forest               │
│ Support Vector Machine      │
│ XGBoost                     │
└─────────────────────────────┘
       ↓
Phishing / Legitimate Prediction
       ↓
Results displayed on Web Interface
```

## ✨ Features

- 🔍 URL-based phishing detection
- 🧠 Three machine learning models
- 📊 59 numerical URL features
- 🌐 Flask web interface
- ⚡ Real-time URL prediction
- 🔄 Model comparison
- 📱 Responsive cybersecurity-themed interface

## 🤖 Machine Learning Algorithms

### 1. Random Forest

Random Forest uses multiple decision trees and combines their predictions to make the final classification.

**Simple idea:**  
Many decision trees → voting → final prediction

### 2. Support Vector Machine (SVM)

SVM finds a decision boundary that separates different classes in the feature space.

**Simple idea:**  
Find the best boundary → classify the URL

### 3. XGBoost

XGBoost is a gradient boosting algorithm where multiple decision trees are built sequentially to improve prediction performance.

**Simple idea:**  
Build trees → learn from previous errors → improve prediction

## 🔢 URL Features

The system extracts 59 numerical features from a URL.

Examples include:

- URL length
- Hostname length
- IP address usage
- Number of dots
- Number of hyphens
- Number of special characters
- HTTPS usage
- Digit count
- Letter count
- Path length
- Query length
- Subdomain count
- TLD length
- URL entropy
- Hostname entropy
- Suspicious keyword count
- Suspicious TLD flag
- Digit ratio
- Letter ratio
- Path digit ratio
- Hostname character frequency

These features are generated directly from the URL, allowing the system to make predictions without requiring the complete webpage.

## 📂 Project Structure

```text
PhishGuard-AI/
│
├── dataset/
│
├── models/
│
├── app.py
├── feature_extraction.py
├── train_model.py
├── test_prediction.py
├── dataset_check.py
├── domain_evaluation.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

Dataset CSV files and trained model files are excluded from Git using `.gitignore`.

## 📊 Dataset

This project uses the **PhiUSIIL Phishing URL Dataset**.

The dataset contains **235,795 URL records** with phishing/legitimate labels.

### Dataset Labels

```text
1 → Legitimate
0 → Phishing
```

## 🧪 Model Performance

The models were evaluated using an 80/20 stratified train-test split.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Random Forest | 99.80% | 99.66% | 99.99% | 99.82% |
| SVM | 99.76% | 99.62% | 99.97% | 99.79% |
| XGBoost | 99.79% | 99.65% | 99.99% | 99.81% |

These results are based on the project's dataset and evaluation setup. They should not be interpreted as a guarantee of real-world phishing detection accuracy.

## 🌐 Web Application

The project uses **Flask** to provide a web interface.

The user enters a URL, and the application:

1. Receives the URL
2. Extracts 59 numerical features
3. Loads the trained models
4. Sends the features to Random Forest, SVM and XGBoost
5. Displays the prediction of each model
6. Shows the overall model consensus

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PhishGuard-AI.git
cd PhishGuard-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

Start the Flask application:

```bash
python app.py
```

Then open the local address shown by Flask in your browser.

## 🧠 Training the Models

The training script extracts URL features and trains the three machine learning models.

```bash
python train_model.py
```

The trained models are saved in the `models/` directory.

Because the dataset and trained model files are excluded from GitHub, they must be available locally before running the complete application.

## 📌 Limitations

- The system analyzes URL-based features only.
- It does not inspect complete webpage HTML.
- Model performance depends on the training dataset.
- New phishing techniques may not always be detected.
- Predictions should not be treated as a security guarantee.
- External validation on completely independent datasets would be useful.

## 🔮 Future Scope

Possible future improvements include:

- HTML and webpage-content analysis
- Domain reputation checking
- WHOIS information
- DNS and SSL-based features
- Threat intelligence integration
- Larger and more diverse datasets
- Explainable AI
- Browser extension
- Real-time phishing intelligence
- Independent external dataset evaluation

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Flask
- Matplotlib
- Seaborn
- Git & GitHub

## 👨‍💻 Project Type

**Academic Machine Learning Project**

**Project:** PhishGuard AI  
**Topic:** Phishing Website Detection Using Machine Learning  
**Interface:** Flask Web Application  
**Algorithms:** Random Forest, SVM, XGBoost

## ⚠️ Disclaimer

PhishGuard AI is an academic machine learning project intended for educational and research purposes.

A prediction from this system should not be considered a definitive security verdict. Users should use additional security tools and safe browsing practices when evaluating suspicious websites.