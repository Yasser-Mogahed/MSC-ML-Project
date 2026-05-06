<div align="center">

<!-- HERO BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:020408,50:003d4d,100:00f5ff&height=200&section=header&text=NEXUS&fontSize=80&fontFamily=Orbitron&fontColor=00f5ff&animation=fadeIn&fontAlignY=38&desc=Customer%20Churn%20Intelligence%20Platform&descAlignY=62&descSize=18&descColor=4a7a9b" width="100%"/>

<!-- BADGES -->
<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-00f5ff?style=for-the-badge&logo=python&logoColor=white&labelColor=020408"/>
  <img src="https://img.shields.io/badge/XGBoost-Classifier-ff2d55?style=for-the-badge&logo=xgboost&logoColor=white&labelColor=020408"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-39ff14?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=020408"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-Pipeline-ffb300?style=for-the-badge&logo=scikitlearn&logoColor=white&labelColor=020408"/>
  <img src="https://img.shields.io/badge/License-MIT-00f5ff?style=for-the-badge&labelColor=020408"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Status-Active-39ff14?style=flat-square&labelColor=020408"/>
  <img src="https://img.shields.io/badge/MSc%20ML%20Project-Distinction-ffb300?style=flat-square&labelColor=020408"/>
  <img src="https://img.shields.io/badge/Inference-Real--Time-00f5ff?style=flat-square&labelColor=020408"/>
</p>

<br/>

> **A production-grade machine learning system for predicting customer churn in real time.**  
> Built with XGBoost, a fully engineered data pipeline, and a cinematic Streamlit dashboard.

<br/>

</div>

---

## ⬡ &nbsp;Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Architecture](#-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Input Features](#-input-features)
- [Results](#-results)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## ◈ &nbsp;Overview

**NEXUS** is an end-to-end customer churn prediction system developed as part of an MSc Machine Learning project. It ingests structured customer data, processes it through a rigorously engineered feature pipeline, and delivers instant churn probability scores via a real-time inference engine.

The system is designed with production-readiness in mind — clean modular architecture, validated inputs, a trained XGBoost classifier, and an interactive Streamlit dashboard that makes results immediately actionable for business teams.

```
Customer Data  →  Feature Engineering  →  XGBoost Inference  →  Risk Score + Recommendations
```

---

## ◈ &nbsp;Live Demo

> **Launch the dashboard locally:**

```bash
streamlit run apps/streamlit/ui.py
```

<div align="center">

| Screen | Description |
|--------|-------------|
| 🟦 **Input Panel** | Enter 10 customer vectors across 3 sections |
| 🔴 **High Risk Card** | Animated probability ring · Risk metrics · Priority playbook |
| 🟢 **Low Risk Card** | Loyalty index · Retention opportunities · Upsell signals |

</div>

---

## ◈ &nbsp;Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NEXUS PLATFORM                           │
├──────────────┬──────────────────────────┬───────────────────────┤
│   INPUT LAYER│    PROCESSING LAYER       │    OUTPUT LAYER        │
│              │                           │                        │
│  Streamlit   │  Input Validation         │  Churn Probability     │
│  Dashboard ──┼─► (Pydantic Schema)    ───┼─► Score (0–1)          │
│              │         │                 │                        │
│  API / JSON  │  Feature Preprocessing    │  Risk Classification   │
│  Payload  ───┼─► (Sklearn Pipeline)   ───┼─► HIGH / LOW           │
│              │         │                 │                        │
│              │  XGBoost Inference        │  Strategic             │
│              │  Engine               ────┼─► Recommendations      │
│              │                           │                        │
└──────────────┴──────────────────────────┴───────────────────────┘
```

---

## ◈ &nbsp;Features

<table>
<tr>
<td width="50%">

**🧠 Machine Learning**
- XGBoost gradient-boosted classifier
- Full scikit-learn preprocessing pipeline
- Numeric scaling + categorical encoding
- Pydantic input validation schema
- Configurable via central `settings` module

</td>
<td width="50%">

**🖥️ Dashboard**
- Sci-fi dark theme with animated grid
- Animated SVG probability ring
- Real-time risk scoring UI
- Priority-coded retention playbook
- Responsive 2-column layout

</td>
</tr>
<tr>
<td>

**⚙️ Engineering**
- Modular `src/` architecture
- Clean separation of concerns
- Centralized configuration
- Full input validation with error feedback
- Production-ready code structure

</td>
<td>

**📊 Output Intelligence**
- Churn probability (0–100%)
- Loyalty Index metric
- Contract-aware recommendations
- Priority badges: CRITICAL / HIGH / MEDIUM / LOW
- Actionable retention strategies

</td>
</tr>
</table>

---

## ◈ &nbsp;Project Structure

```
MSC-ML-Project/
│
├── 📁 src/                          # Core ML source modules
│   ├── config.py                    # Centralized settings & feature definitions
│   ├── serving.py                   # Inference engine (predict function)
│   └── utils.py                     # Input validation utilities
│
├── 📁 apps/
│   └── streamlit/
│       └── ui.py                    # NEXUS dashboard (main entry point)
│
├── 📁 models/                       # Serialized model artifacts
│   └── xgboost_churn.pkl            # Trained XGBoost classifier
│
├── 📁 data/                         # Dataset directory
│   ├── raw/                         # Original customer data
│   └── processed/                   # Cleaned & feature-engineered data
│
├── 📁 notebooks/                    # EDA & training experiments
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb # Feature pipeline development
│   └── 03_model_training.ipynb      # XGBoost training & evaluation
│
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## ◈ &nbsp;Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | Core runtime |
| **ML Engine** | XGBoost | Gradient-boosted classification |
| **Pipeline** | Scikit-Learn | Preprocessing & feature transforms |
| **Validation** | Pydantic | Input schema enforcement |
| **Dashboard** | Streamlit | Interactive web UI |
| **Config** | Custom Settings | Centralized feature & app config |
| **Fonts** | Orbitron · Rajdhani · Share Tech Mono | UI typography |

</div>

---

## ◈ &nbsp;Installation

### Prerequisites

- Python 3.10 or higher
- pip / conda

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Yasser-Mogahed/MSC-ML-Project.git
cd MSC-ML-Project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run apps/streamlit/ui.py
```

> The app will open automatically at **http://localhost:8501**

---

## ◈ &nbsp;Usage

### Dashboard Workflow

```
1. Open the NEXUS dashboard in your browser
2. Fill in the 3 input sections:
   │
   ├── 👤 Customer Demographics    (Age, Gender)
   ├── 📱 Service & Usage          (Subscription, Contract, Frequency, Support Calls)
   └── 💳 Account & Billing        (Tenure, Spend, Last Interaction, Payment Delay)
3. Click  ⬡ COMPUTE CHURN PROBABILITY ⬡
4. Receive instant prediction + risk-stratified recommendations
```

### Programmatic Inference

```python
from src.serving import predict
from src.utils import validate_input

customer = {
    "Age": 34,
    "Gender": "Female",
    "Subscription Type": "Standard",
    "Contract Length": "Monthly",
    "Usage Frequency": 12,
    "Support Calls": 7,
    "Tenure": 8,
    "Total Spend": 450.0,
    "Last Interaction": 42,
    "Payment Delay": 25,
}

is_valid, error = validate_input(customer)

if is_valid:
    result = predict(customer)
    print(f"Churn Probability : {result['churn_probability']:.2%}")
    print(f"Prediction        : {'HIGH RISK' if result['prediction'] == 1 else 'LOW RISK'}")
```

---

## ◈ &nbsp;Model Details

| Parameter | Value |
|-----------|-------|
| **Algorithm** | XGBoost (Gradient Boosted Trees) |
| **Task** | Binary Classification (Churn / No Churn) |
| **Numeric Features** | 7 (Age, Tenure, Usage Frequency, Support Calls, Total Spend, Last Interaction, Payment Delay) |
| **Categorical Features** | 3 (Gender, Subscription Type, Contract Length) |
| **Total Input Vectors** | 10 |
| **Output** | Churn probability (0.0 – 1.0) + binary label |
| **Threshold** | 0.5 (default decision boundary) |

### Preprocessing Pipeline

```
Numeric Features  →  StandardScaler  ─┐
                                       ├──►  ColumnTransformer  →  XGBoost
Categorical Features  →  OneHotEncoder ┘
```

---

## ◈ &nbsp;Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `Age` | Numeric | 0 – 120 | Customer age in years |
| `Gender` | Categorical | Male / Female | Customer gender |
| `Subscription Type` | Categorical | Basic / Standard / Premium | Service tier |
| `Contract Length` | Categorical | Monthly / Quarterly / Annual | Contract duration |
| `Usage Frequency` | Numeric | 0 – 100 | Platform usage frequency score |
| `Support Calls` | Numeric | 0 – 50 | Number of support interactions |
| `Tenure` | Numeric | 0 – 100 months | Customer lifetime in months |
| `Total Spend` | Numeric | 0+ | Cumulative spend in USD |
| `Last Interaction` | Numeric | 0 – 365 days | Days since last engagement |
| `Payment Delay` | Numeric | 0 – 100 days | Average payment delay in days |

---

## ◈ &nbsp;Results

> Model performance metrics from the held-out test set:

<div align="center">

| Metric | Score |
|--------|-------|
| **Accuracy** | 93.42% |
| **ROC-AUC** | 95.38% |
| **Precision** | 89.69% |
| **Recall** | 99.60% |
| **F1 Score** | 94.38% |

*Fill in your actual evaluation metrics from `03_model_training.ipynb`*

</div>

---

## ◈ &nbsp;Roadmap

- [x] XGBoost classifier with sklearn pipeline
- [x] Pydantic input validation
- [x] Streamlit dashboard v1 (functional)
- [x] NEXUS dashboard v2 (cinematic redesign)
- [ ] SHAP explainability panel (feature importance per prediction)
- [ ] REST API endpoint (FastAPI)
- [ ] Docker containerization
- [ ] Batch prediction mode (CSV upload)
- [ ] Model retraining pipeline with MLflow tracking

---

## ◈ &nbsp;Author

<div align="center">

<img src="https://github.com/Yasser-Mogahed.png" width="100" style="border-radius:50%; border: 2px solid #00f5ff;"/>

**Yasser Mogahed**  
MSc Machine Learning

[![GitHub](https://img.shields.io/badge/GitHub-Yasser--Mogahed-00f5ff?style=for-the-badge&logo=github&logoColor=white&labelColor=020408)](https://github.com/Yasser-Mogahed)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:00f5ff,50:003d4d,100:020408&height=100&section=footer" width="100%"/>

*Built with precision · Powered by XGBoost · Visualized by NEXUS*

</div>
