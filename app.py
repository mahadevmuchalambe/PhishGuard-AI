from flask import Flask, request, render_template_string
import joblib
import pandas as pd
from feature_extraction import extract_url_features

app = Flask(__name__)

# ==========================================================
# LOAD TRAINED MODELS
# ==========================================================

RF_MODEL_PATH = "models/random_forest_model.pkl"
SVM_MODEL_PATH = "models/svm_model.pkl"
XGB_MODEL_PATH = "models/xgboost_model.pkl"

random_forest = joblib.load(RF_MODEL_PATH)
svm_model = joblib.load(SVM_MODEL_PATH)
xgboost_model = joblib.load(XGB_MODEL_PATH)


# ==========================================================
# HTML + CSS
# ==========================================================

HTML = """

<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>PhishGuard AI</title>

<style>

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background:
        radial-gradient(
            circle at top,
            #172554 0%,
            #0f172a 35%,
            #020617 75%
        );

    color: #e2e8f0;

    min-height: 100vh;

}


/* ==========================================================
   BACKGROUND GRID
========================================================== */

body::before {

    content: "";

    position: fixed;

    inset: 0;

    pointer-events: none;

    background-image:
        linear-gradient(
            rgba(59,130,246,0.035) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(59,130,246,0.035) 1px,
            transparent 1px
        );

    background-size: 40px 40px;

}


/* ==========================================================
   NAVBAR
========================================================== */

.navbar {

    width: 100%;

    padding: 22px 7%;

    display: flex;

    justify-content: space-between;

    align-items: center;

    border-bottom:
        1px solid rgba(148,163,184,0.12);

    background:
        rgba(2,6,23,0.55);

    backdrop-filter: blur(12px);

}


.logo {

    display: flex;

    align-items: center;

    gap: 12px;

}


.logo-icon {

    width: 42px;

    height: 42px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 12px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #06b6d4
        );

    box-shadow:
        0 0 30px rgba(37,99,235,0.35);

    font-size: 20px;

}


.logo-text {

    font-size: 20px;

    font-weight: 800;

    letter-spacing: -0.5px;

}


.logo-text span {

    color: #38bdf8;

}


.status {

    display: flex;

    align-items: center;

    gap: 8px;

    color: #94a3b8;

    font-size: 13px;

}


.status-dot {

    width: 8px;

    height: 8px;

    border-radius: 50%;

    background: #22c55e;

    box-shadow:
        0 0 12px rgba(34,197,94,0.8);

}


/* ==========================================================
   MAIN
========================================================== */

.container {

    width: 86%;

    max-width: 1200px;

    margin: auto;

}


/* ==========================================================
   HERO
========================================================== */

.hero {

    text-align: center;

    padding:
        75px 20px 35px;

}


.badge {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 8px 15px;

    border-radius: 999px;

    border:
        1px solid rgba(56,189,248,0.25);

    background:
        rgba(14,165,233,0.08);

    color: #7dd3fc;

    font-size: 12px;

    font-weight: 700;

    letter-spacing: 0.5px;

    text-transform: uppercase;

}


.hero h1 {

    margin-top: 22px;

    font-size:
        clamp(38px, 6vw, 68px);

    line-height: 1.05;

    letter-spacing: -2px;

    color: #f8fafc;

}


.gradient-text {

    background:
        linear-gradient(
            90deg,
            #38bdf8,
            #818cf8,
            #c084fc
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

}


.hero p {

    max-width: 680px;

    margin:
        20px auto 0;

    color: #94a3b8;

    font-size: 16px;

    line-height: 1.7;

}


/* ==========================================================
   ANALYZER
========================================================== */

.analyzer {

    margin-top: 35px;

    padding: 8px;

    border-radius: 18px;

    background:
        linear-gradient(
            135deg,
            rgba(56,189,248,0.35),
            rgba(129,140,248,0.25),
            rgba(192,132,252,0.25)
        );

    box-shadow:
        0 20px 70px rgba(0,0,0,0.35);

}


.analyzer-inner {

    background: #0f172a;

    border-radius: 13px;

    padding: 10px;

}


form {

    display: flex;

    gap: 10px;

}


.url-input {

    flex: 1;

    background: #020617;

    border:
        1px solid #1e293b;

    border-radius: 11px;

    padding: 18px;

    color: #f8fafc;

    font-size: 15px;

    outline: none;

    transition: 0.25s;

}


.url-input:focus {

    border-color: #38bdf8;

    box-shadow:
        0 0 0 3px rgba(56,189,248,0.08);

}


.url-input::placeholder {

    color: #64748b;

}


.analyze-btn {

    border: none;

    border-radius: 11px;

    padding:
        0 28px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );

    color: white;

    font-size: 14px;

    font-weight: 800;

    cursor: pointer;

    transition:
        transform 0.2s,
        box-shadow 0.2s;

}


.analyze-btn:hover {

    transform: translateY(-2px);

    box-shadow:
        0 10px 30px rgba(99,102,241,0.35);

}


/* ==========================================================
   TECH INFO
========================================================== */

.tech-row {

    display: flex;

    justify-content: center;

    flex-wrap: wrap;

    gap: 10px;

    margin-top: 18px;

}


.tech {

    padding: 7px 12px;

    border-radius: 7px;

    background:
        rgba(15,23,42,0.8);

    border:
        1px solid #1e293b;

    color: #64748b;

    font-size: 11px;

    font-family: monospace;

}


/* ==========================================================
   RESULTS
========================================================== */

.results {

    margin-top: 45px;

}


.result-header {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 18px;

}


.result-header h2 {

    font-size: 20px;

}


.scan-status {

    font-size: 12px;

    color: #64748b;

    font-family: monospace;

}


/* ==========================================================
   MODEL GRID
========================================================== */

.models {

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 16px;

}


.model-card {

    position: relative;

    padding: 24px;

    background:
        rgba(15,23,42,0.72);

    border:
        1px solid #1e293b;

    border-radius: 16px;

    overflow: hidden;

    transition: 0.25s;

}


.model-card:hover {

    transform: translateY(-4px);

    border-color:
        rgba(56,189,248,0.35);

    box-shadow:
        0 15px 40px rgba(0,0,0,0.25);

}


.model-card::before {

    content: "";

    position: absolute;

    top: 0;

    left: 0;

    width: 100%;

    height: 2px;

    background:
        linear-gradient(
            90deg,
            #38bdf8,
            #818cf8
        );

}


.model-name {

    color: #94a3b8;

    font-size: 13px;

    font-weight: 600;

}


.prediction {

    margin-top: 15px;

    font-size: 27px;

    font-weight: 900;

}


.phishing {

    color: #fb7185;

}


.legitimate {

    color: #4ade80;

}


.model-type {

    margin-top: 8px;

    color: #475569;

    font-size: 11px;

    font-family: monospace;

}


/* ==========================================================
   SUMMARY
========================================================== */

.summary {

    margin-top: 18px;

    padding: 25px;

    border-radius: 16px;

    background:
        linear-gradient(
            135deg,
            rgba(30,41,59,0.8),
            rgba(15,23,42,0.8)
        );

    border:
        1px solid #1e293b;

}


.summary-label {

    color: #64748b;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 1px;

}


.summary-result {

    margin-top: 8px;

    font-size: 24px;

    font-weight: 900;

}


.summary-detail {

    margin-top: 7px;

    color: #94a3b8;

    font-size: 13px;

}


/* ==========================================================
   DISCLAIMER
========================================================== */

.disclaimer {

    margin-top: 22px;

    padding: 16px;

    border-radius: 10px;

    background:
        rgba(245,158,11,0.06);

    border:
        1px solid rgba(245,158,11,0.15);

    color: #94a3b8;

    font-size: 12px;

    line-height: 1.6;

}


/* ==========================================================
   FOOTER
========================================================== */

footer {

    text-align: center;

    padding: 45px 20px;

    color: #475569;

    font-size: 12px;

}


footer span {

    color: #38bdf8;

}


/* ==========================================================
   RESPONSIVE
========================================================== */

@media (max-width: 800px) {

    .models {

        grid-template-columns: 1fr;

    }

    form {

        flex-direction: column;

    }

    .analyze-btn {

        padding: 17px;

    }

    .navbar {

        padding:
            18px 5%;

    }

    .container {

        width: 92%;

    }

}

</style>

</head>


<body>


<!-- ========================================================
     NAVBAR
======================================================== -->

<nav class="navbar">

    <div class="logo">

        <div class="logo-icon">
            🛡️
        </div>

        <div class="logo-text">
            Phish<span>Guard</span> AI
        </div>

    </div>


    <div class="status">

        <div class="status-dot"></div>

        ML Engine Online

    </div>

</nav>


<!-- ========================================================
     MAIN
======================================================== -->

<main class="container">


<section class="hero">

    <div class="badge">
        ⚡ Machine Learning Security Engine
    </div>


    <h1>

        Detect
        <span class="gradient-text">
            Phishing URLs
        </span>

        with AI

    </h1>


    <p>

        Analyze suspicious website URLs using
        machine learning and multiple classification
        algorithms to identify potential phishing threats.

    </p>


    <!-- ANALYZER -->

    <div class="analyzer">

        <div class="analyzer-inner">

            <form method="POST">

                <input
                    class="url-input"
                    type="text"
                    name="url"
                    value="{{ url }}"
                    placeholder="https://example.com/login"
                    required
                >

                <button
                    class="analyze-btn"
                    type="submit">

                    Analyze URL →

                </button>

            </form>

        </div>

    </div>


    <div class="tech-row">

        <div class="tech">
            59 URL FEATURES
        </div>

        <div class="tech">
            RANDOM FOREST
        </div>

        <div class="tech">
            SVM
        </div>

        <div class="tech">
            XGBOOST
        </div>

    </div>

</section>


{% if results %}


<!-- ========================================================
     RESULTS
======================================================== -->

<section class="results">


    <div class="result-header">

        <h2>
            Analysis Results
        </h2>

        <div class="scan-status">
            SCAN_COMPLETE
        </div>

    </div>


    <!-- MODEL CARDS -->

    <div class="models">


        {% for model, prediction in results.items() %}


        <div class="model-card">

            <div class="model-name">
                {{ model }}
            </div>


            <div class="prediction
                {% if prediction == 'Phishing' %}
                    phishing
                {% else %}
                    legitimate
                {% endif %}
            ">

                {% if prediction == "Phishing" %}
                    ⚠ Phishing
                {% else %}
                    ✓ Legitimate
                {% endif %}

            </div>


            <div class="model-type">

                {% if model == "Random Forest" %}
                    ensemble_classifier
                {% elif model == "SVM" %}
                    linear_classifier
                {% else %}
                    gradient_boosting
                {% endif %}

            </div>

        </div>


        {% endfor %}


    </div>


    <!-- SUMMARY -->

    <div class="summary">

        <div class="summary-label">
            Model Consensus
        </div>


        {% set phishing_count =
            results.values() | select("equalto", "Phishing") | list | length
        %}


        {% if phishing_count == 3 %}

            <div class="summary-result phishing">
                ⚠ Potential Phishing Detected
            </div>

            <div class="summary-detail">
                All three machine learning models classified
                this URL as phishing.
            </div>

        {% elif phishing_count == 0 %}

            <div class="summary-result legitimate">
                ✓ No Phishing Signal Detected
            </div>

            <div class="summary-detail">
                All three machine learning models classified
                this URL as legitimate.
            </div>

        {% else %}

            <div class="summary-result">
                ◈ Models Disagree
            </div>

            <div class="summary-detail">
                The machine learning models produced different
                predictions. Treat this result with caution.
            </div>

        {% endif %}

    </div>


    <!-- DISCLAIMER -->

    <div class="disclaimer">

        <strong>Important:</strong>
        This system provides an ML-based prediction using
        URL characteristics. It is not a guarantee that a
        website is safe or malicious. Always verify suspicious
        links before interacting with them.

    </div>


</section>


{% endif %}


</main>


<footer>

    Built with
    <span>Flask</span> •
    <span>Scikit-learn</span> •
    <span>XGBoost</span>

    <br><br>

    PhishGuard AI — Machine Learning Based
    Phishing URL Detection

</footer>


</body>

</html>

"""


# ==========================================================
# HOME ROUTE
# ==========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    results = None
    url = ""

    if request.method == "POST":

        url = request.form.get("url", "").strip()

        if url:

            # Extract 59 URL features
            features = extract_url_features(url)

            # Convert features into DataFrame
            features_df = pd.DataFrame([features])

            # Predictions
            rf_prediction = random_forest.predict(features_df)[0]
            svm_prediction = svm_model.predict(features_df)[0]
            xgb_prediction = xgboost_model.predict(features_df)[0]

            # 1 = Legitimate
            # 0 = Phishing

            results = {

                "Random Forest":
                    "Legitimate"
                    if rf_prediction == 1
                    else "Phishing",

                "SVM":
                    "Legitimate"
                    if svm_prediction == 1
                    else "Phishing",

                "XGBoost":
                    "Legitimate"
                    if xgb_prediction == 1
                    else "Phishing"
            }

    return render_template_string(
        HTML,
        results=results,
        url=url
    )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(debug=True)