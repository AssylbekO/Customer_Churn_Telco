# Telco Customer Churn — Predicting & Profiling

> Machine Learning I — VSE Prague, Summer Semester 2026  
> **Authors:** Anna Kopecny · Adil Zhumagaliyev · Alisha Utegenova · Assylbek Omarov  
> **Supervisor:** prof. Ing. Tomáš Kliegr, Ph.D.

>View the rendered Supervised Notebook here: https://nbviewer.org/github/AssylbekO/Customer_Churn_Telco/blob/main/notebooks/Main_Supervised.ipynb

>View the rendered Unsupervised Notebook here: https://nbviewer.org/github/AssylbekO/Customer_Churn_Telco/blob/main/notebooks/Unsupervised.ipynb

>Results: **0.8414 ROC-AUC · 3 customer segments · 63% cost reduction vs baseline**

---

## Project Overview

Customer churn — the decision of a customer to discontinue their subscription — is one of the most costly problems in the telecommunications industry. Acquiring a new customer costs 5–7× more than retaining an existing one, making early identification of at-risk customers a high-value business objective.

This project analyses the IBM Telco Customer Churn dataset (7,043 customers) across two complementary components:

- **Supervised Learning** — who will churn? Binary classification pipeline optimised for ROC-AUC with cost-sensitive evaluation (FN = 5 × FP)
- **Unsupervised Learning** — what kinds of customers exist? Segmentation via K-Prototypes and Agglomerative Clustering (Gower distance)

---

## Repository Structure

```
telco-churn/
├── data/
│   ├── raw/                        # Original dataset (not tracked by git)
│   └── processed/                  # Train/test splits
├── notebooks/
│   ├── Main_Supervised.ipynb       # Classification pipeline
│   └── Unsupervised.ipynb          # Clustering analysis
├── outputs/
│   ├── Supervised.html             # Rendered notebook
│   ├── Unsupervised.html           # Rendered notebook
│   └── MLI_Customer_Churn.pdf      # Project presentation slides
├── src/
│   └── preprocessor.py             # Custom sklearn Preprocessor class
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Data

**Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Stat | Value |
|---|---|
| Customers | 7,043 |
| Features | 21 attributes + 1 binary target |
| Churn rate (train) | 26.5% |
| Train / Test split | 80/20 stratified → 5,634 / 1,409 |

Feature families: demographics · services · contract · billing

> Raw data is excluded from version control. Download from Kaggle and place in `data/raw/`.

---

## Methods

### Supervised Learning

**Preprocessing**
- Drop CustomerID; coerce TotalCharges (11 blanks → 0)
- Stratified 80/20 split; EDA on training set only
- Custom `Preprocessor`: StandardScaler · OneHotEncoder · OrdinalEncoder (Contract)
- Resampling (ROS / RUS / SMOTENC) treated as a hyperparameter inside `ImbPipeline` — chosen by data, not pre-decided
- All transformations fit on training folds only — no leakage

**Models compared**
- Dummy (majority-class baseline)
- Decision Tree
- Random Forest ← champion
- Neural Network (MLP)

**Evaluation**
- Primary: ROC-AUC (ranking quality)
- Secondary: F1 at tuned threshold (DT 0.49 · RF 0.43 · MLP 0.52)
- Operational: cost matrix with FN = 5, FP = 1

**Interpretability** (on highest-confidence churner, p = 0.881)
- SHAP global feature impact + local waterfall
- LIME tabular explanation
- Anchor rule (98.9% precision): `IF tenure ≤ 9 ∧ PaymentMethod = Electronic check ∧ InternetService = Fiber optic ∧ TotalCharges ≤ 402.98 → Churn`

### Unsupervised Learning

K-Means was ruled out — 16 of 21 features are categorical; one-hot encoding distorts Euclidean distance and yields uninterpretable centroids.

- **K-Prototypes** — Euclidean (numeric) + simple-matching (categorical); centroids = means + modes
- **Agglomerative Clustering** — Gower distance, average linkage; dendrogram for k-selection
- Both algorithms independently selected **k = 3**

---

## Results

### Supervised — Model Comparison (test set, n = 1,409)

| Model | ROC-AUC | F1 | Recall | Cost (5·FN + 1·FP) |
|---|---|---|---|---|
| Baseline (Dummy) | 0.5000 | 0.0000 | 0.0000 | 1,870 |
| Decision Tree | 0.8187 | 0.6054 | 0.7139 | 776 |
| **Random Forest** | **0.8414** | **0.6315** | **0.7513** | **700** |
| Neural Network | 0.8380 | 0.6130 | 0.7433 | 735 |

**Best RF config:** `n_estimators=200, max_depth=7, max_features='sqrt', RUS@0.6, threshold=0.43`

**63% cost reduction** vs baseline ($1,870 → $700) · **75.1% recall** — catches 281 of 374 actual churners

### Unsupervised — Three Customer Segments

| Segment | Size | Churn Rate | Profile |
|---|---|---|---|
| High-Risk Churners | 3,209 | **45.5%** | Tenure ~15 mo · $71/mo · 89% month-to-month · 57% fiber optic · ~80% no add-ons |
| Loyal Long-Term (Budget) | 1,645 | 7.5% | ~$22/mo · 0% fiber · 93% no internet · stable phone-only |
| Loyal High-Value | 2,189 | 13.1% | Tenure ~59 mo · $88/mo · 79% multi-year contracts · 58–72% add-on adoption |

**Key insight:** High-Risk pays only $17/mo less than High-Value yet churns 3.5× more — bundling, not price, is the lock-in mechanism.

**Cross-method agreement:** 85.9% of customers placed in the same segment by both algorithms (Silhouette: KP 0.347 · HC 0.381).

---

## Key Findings

Both supervised and unsupervised analyses converge on the same drivers: **Contract type, tenure, InternetService, add-on bundling.**

| Segment | Strategy | Actions |
|---|---|---|
| High-Risk (Cluster 0) | Contract conversion | Incentives in months 6–18; bundle TechSupport/OnlineSecurity; move from e-check to autopay |
| Loyal High-Value (Cluster 2) | Renewal-window protection | Proactive offers before contract end; loyalty perks for 5+ year customers |
| Loyal Budget (Cluster 1) | Long-term upsell | Low-cost internet upsell; DSL bundle trials; minimal pressure |

---

## Limitations & Future Work

**Limitations**
- Predictive ceiling at ~0.84 ROC-AUC — likely a feature limit, not a tuning failure
- Static snapshot — no time-series view of customer trajectory
- Fiber optic is a confounded signal (price vs quality vs expectations)
- Cluster boundaries are genuinely fuzzy (silhouette 0.35–0.38)

**Future Work**
- Gradient boosting (XGBoost / LightGBM) to test the 0.84 ceiling
- Survival analysis (Cox PH) — predict *when*, not just *if*
- Causal inference on Contract conversion via propensity scoring
- Productionise: scoring API + dashboard with segment-aware retention plays

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/AssylbekO/Customer_Churn_Telco.git
cd Customer_Churn_Telco

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the data
# Download WA_Fn-UseC_-Telco-Customer-Churn.csv from Kaggle → data/raw/

# 4. Run notebooks
jupyter notebook notebooks/Main_Supervised.ipynb
jupyter notebook notebooks/Unsupervised.ipynb
```
