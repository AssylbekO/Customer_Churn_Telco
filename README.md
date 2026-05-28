# Telco Customer Churn Analysis

> Machine Learning I — VSE Prague, Summer Semester 2026  
> **Authors:** Anna Kopecny · Adil Zhumagaliyev · Alisha Utegenova · Assylbek Omarov  
> **Supervisor:** prof. Ing. Tomáš Kliegr, Ph.D.

---

## Project Overview

Customer churn — the decision of a customer to discontinue their subscription — is one of the most costly problems in the telecommunications industry. Acquiring a new customer is estimated to cost five to seven times more than retaining an existing one, making early identification of at-risk customers a high-value business objective.

This project analyses the IBM Telco Customer Churn dataset (7,043 customers) across two components:

- **Supervised Learning** — binary classification pipeline to predict which customers will churn, optimised for ROC-AUC with threshold-tuned F1 reported as a secondary metric
- **Unsupervised Learning** — customer segmentation using K-Prototypes and Agglomerative Clustering to identify distinct behavioural groups

---

## Repository Structure

```
telco-churn/
├── data/
│   ├── raw/                        # Original dataset (not tracked by git)
│   │   └── telco_churn.csv
│   └── processed/                  # Cleaned/split data
│       ├── train.csv
│       └── test.csv
├── notebooks/
│   ├── Supervised.ipynb            # Classification pipeline (main deliverable)
│   └── Unsupervised.ipynb          # Clustering analysis
├── outputs/
│   ├── Supervised.html             # Rendered notebook (GitHub Pages)
│   └── Unsupervised.html           # Rendered notebook (GitHub Pages)
├── src/
│   └── preprocessor.py             # Custom sklearn Preprocessor class
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Data

**Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains 7,043 customer records with 21 features covering:

| Category | Features |
|---|---|
| Demographics | Gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, InternetService, StreamingTV, StreamingMovies, ... |
| Account | Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, tenure |
| Target | `Churn` — binary (1 = churned, 0 = retained) |

> Raw data is excluded from version control via `.gitignore`. Download from Kaggle and place in `data/raw/`.

---

## Methods

### Supervised Learning
- **EDA** — six-phase exploratory framework; outlier detection via IQR and Isolation Forest (`contamination=0.01`)
- **Preprocessing** — custom `Preprocessor` class (inherits `BaseEstimator`, `TransformerMixin`); OneHotEncoding, OrdinalEncoding, StandardScaler
- **Class imbalance** — SMOTENC via `ImbPipeline`
- **Models** — Dummy baseline, Decision Tree, Random Forest, MLP; tuned with `GridSearchCV` + `StratifiedKFold`
- **Evaluation** — ROC-AUC (primary), F1 at tuned threshold, cost-sensitive confusion matrix (FN cost = 5, FP cost = 1)
- **Interpretability** — SHAP, LIME, Anchor explanations on the highest-confidence churn prediction

### Unsupervised Learning
- **Distance** — Gower distance for mixed-type features
- **Models** — K-Prototypes, Agglomerative Clustering (Ward linkage)
- **Validation** — Elbow method (KneeLocator), Silhouette score, Dendrogram inspection
- **Profiling** — Chi-squared tests and group statistics per cluster

---

## Results

| Model | ROC-AUC | F1 (tuned threshold) |
|---|---|---|
| Dummy (baseline) | ~0.50 | — |
| Decision Tree | TBD | TBD |
| Random Forest | TBD | TBD |
| MLP | TBD | TBD |

> Results will be updated upon final model evaluation.

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/AssylbekO/telco-churn.git
cd telco-churn
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the data**  
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in `data/raw/`.

**4. Run the notebooks**
```bash
jupyter notebook notebooks/Supervised.ipynb
jupyter notebook notebooks/Unsupervised.ipynb
```

---

## Dependencies

Key libraries used in this project:

| Library | Purpose |
|---|---|
| scikit-learn | Pipelines, models, evaluation |
| imbalanced-learn | SMOTENC oversampling |
| kmodes | K-Prototypes clustering |
| gower | Mixed-type distance matrix |
| shap / lime / anchor-exp | Model interpretability |
| kneed | Elbow point detection |
| graphviz | Decision tree visualisation |
