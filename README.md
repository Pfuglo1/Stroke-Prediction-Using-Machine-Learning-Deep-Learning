# Stroke Prediction Using Machine Learning & Deep Learning
 
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-189AB4)
![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-success)
 
A stroke risk screening project that predicts whether a patient is likely to have a stroke from demographic, lifestyle and clinical information. The project tackles a **severely imbalanced dataset (4.9% positive cases)**. It compares four machine learning models with a deep neural network and uses **decision-threshold tuning** to move recall from **0% to 88%**.
 
---
 
## Table of Contents
 
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Key Insights](#key-insights)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [Author](#author)
---
 
## Problem Statement
 
Stroke is one of the leading causes of death and long-term disability worldwide, and many strokes are preventable when high-risk patients are identified early. A screening model that flags at-risk individuals can help clinicians prioritise preventive care, such as blood-pressure control, glucose management and lifestyle changes.
 
The task is **binary classification**:
 
| Label | Meaning | Share |
|---|---|---|
| `0` | No stroke | 95.1% |
| `1` | **Stroke** | 4.9% (249 patients) |
 
> ⚠️ **The core challenge is extreme imbalance.** A model that always predicts "no stroke" scores 95% accuracy while catching zero strokes. **Recall** on the stroke class is therefore the key metric.
 
---
 
## Dataset
 
**Source:** [Stroke Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
 
- **Patients:** 5,110
- **Features:** 10 + target (`id` dropped)
| Feature | Type | Description |
|---|---|---|
| `gender` | Categorical | Male, Female, Other |
| `age` | Numerical | Age in years (0.08 – 82) |
| `hypertension` | Binary | 1 = has hypertension |
| `heart_disease` | Binary | 1 = has heart disease |
| `ever_married` | Categorical | Yes / No |
| `work_type` | Categorical | Private, Self-employed, Govt_job, Children, Never_worked |
| `Residence_type` | Categorical | Urban / Rural |
| `avg_glucose_level` | Numerical | Average blood glucose (mg/dL) |
| `bmi` | Numerical | Body Mass Index (201 missing values) |
| `smoking_status` | Categorical | Never smoked, Formerly smoked, Smokes, Unknown |
| `stroke` | **Target** | 1 = had a stroke |
 
---
 
## Project Workflow
 
### 1. Data Cleaning
- **Missing values:** `bmi` had 201 missing values (3.9%), imputed with the **median (28.1)** because BMI is right-skewed.
- No duplicate rows.
- Dropped the `id` column.
- Removed a single record with `gender = 'Other'` (one sample is too few to learn from).
### 2. Exploratory Data Analysis
- Class distribution (95.1% / 4.9%)
- Histograms, KDE plots and box plots for `age`, `avg_glucose_level` and `bmi`
- Stroke rate by every categorical feature
- Line plots of key risk factors against stroke
- Pearson and Spearman correlation heatmaps, plus a feature–target correlation chart
**Stroke rate by risk factor**
 
| Factor | Without | With | Relative risk |
|---|---|---|---|
| Heart disease | 4.2% | **17.0%** | ~4× |
| Hypertension | 4.0% | **13.3%** | ~3.3× |
| Ever married | 1.7% | 6.6% | ~3.9× *(age effect)* |
| Smoking: formerly smoked | 4.8% (never) | **7.9%** | ~1.7× |
| Work: self-employed | 5.1% (private) | **7.9%** | ~1.6× |
| Gender (M vs F) | 4.7% (F) | 5.1% (M) | ~1.1× |
| Residence (Urban vs Rural) | 4.5% (Rural) | 5.2% (Urban) | ~1.1× |
 
### 3. Transformation & Encoding
- Applied a **Yeo-Johnson power transform** to `age`, `avg_glucose_level` and `bmi` (glucose skew dropped from 1.57 to 0.08).
- Mapped categorical columns to integers with explicit dictionaries.
### 4. Feature Engineering
Nine clinically motivated features were created:
 
| Feature | Logic |
|---|---|
| `cardio_risk_score` | hypertension + heart_disease + high glucose + current smoker |
| `age_hypertension` | age × hypertension |
| `age_heart_disease` | age × heart_disease |
| `overall_health_risk` | cardio_risk_score + obese + age ≥ 60 |
| `elderly_smoker` | age ≥ 60 **and** current smoker |
| `married_senior` | married **and** age ≥ 60 |
| `glucose_bmi_ratio` | glucose / (bmi + 1) |
| `disease_burden` | hypertension + heart_disease |
| `age_glucose` | age × glucose |
 
The final dataset has **5,109 rows × 19 features + target**.
 
### 5. Modelling
- Stratified **80/20** split (`random_state=42`) and `StandardScaler` fitted on the training data only.
- **ML baselines:** Logistic Regression, Decision Tree, Random Forest, XGBoost.
- **Threshold tuning:** lowered the decision threshold to trade precision for recall.
- **Deep neural network** with `EarlyStopping` on validation loss (patience = 5).
- Saved the trained network (`dl_model.keras`) and scaler (`scaler.pkl`), and verified inference on a sample stroke patient.
---
 
## Model Architecture
 
```
Input (19 features)
   │
Dense(128, ReLU)
   │
Dense(64, ReLU)
   │
Dense(64, ReLU)
   │
Dense(1, Sigmoid)  →  P(stroke)
```
 
**Total trainable parameters:** 15,041 · **Optimiser:** Adam · **Loss:** binary cross-entropy
 
---
 
## Results
 
Test set: **1,022 patients** (972 no stroke, 50 stroke)
 
### Default threshold (0.5): the accuracy trap
 
| Model | Accuracy | Precision | **Recall** | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.950 | 0.000 | **0.00** | 0.000 |
| Random Forest | 0.952 | 0.600 | **0.06** | 0.109 |
| XGBoost | 0.936 | 0.105 | **0.04** | 0.058 |
| Decision Tree | 0.912 | 0.197 | **0.26** | 0.224 |
 
Every model looks good on accuracy (91–95%) but misses most strokes. Logistic Regression catches **none**.
 
### Tuned thresholds: prioritising recall
 
| Model | Threshold | Strokes caught | **Recall** | Precision | Patients flagged |
|---|---|---|---|---|---|
| XGBoost | 0.01 | 37 / 50 | **0.74** | 0.12 | 309 (30%) |
| **Deep Neural Network** | 0.001 | **44 / 50** | **0.88** | 0.08 | 539 (53%) |
 
**Deep learning confusion matrix (threshold 0.001)**
 
|  | Predicted: No stroke | Predicted: Stroke |
|---|---|---|
| **Actual: No stroke** | 477 | 495 |
| **Actual: Stroke** | **6** | 44 |
 
**Takeaway:** Threshold tuning turns a model that misses almost every stroke into one that catches **88%** of them. The trade-off is substantial: about half of healthy patients are also flagged. That can be acceptable for a *first-stage* screening tool followed by cheaper confirmatory checks, but it shows that this dataset needs proper imbalance handling (see [Limitations](#limitations--future-work)).
 
---
 
## Key Insights
 
**Decision Tree feature importance (top 5)**
 
| Rank | Feature | Importance |
|---|---|---|
| 1 | `bmi` | 0.188 |
| 2 | `age` | 0.150 |
| 3 | `glucose_bmi_ratio` *(engineered)* | 0.150 |
| 4 | `avg_glucose_level` | 0.141 |
| 5 | `age_glucose` *(engineered)* | 0.105 |
 
- 👴 **Age** is the dominant risk factor. "Ever married" and "self-employed" look risky mainly because those groups are older.
- ❤️ **Heart disease** (17% stroke rate) and **hypertension** (13%) are the strongest binary risk factors.
- 🩸 **Glucose and BMI**, alone and combined, are among the most informative continuous features.
- 🚻 **Gender and residence type** have almost no effect.
---
 
## Tech Stack
 
| Category | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Preprocessing | scikit-learn (`PowerTransformer`, `StandardScaler`) |
| Machine learning | scikit-learn (Logistic Regression, Decision Tree, Random Forest), XGBoost |
| Deep learning | TensorFlow / Keras |
| Model persistence | Pickle, Keras `.keras` format |
| Environment | Google Colab |
 
---
 
## How to Run
 
### Option 1: Google Colab (recommended)
1. Download `healthcare-dataset-stroke-data.csv` from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
2. Open the notebook in [Google Colab](https://colab.research.google.com/).
3. Upload the CSV to `/content/`.
4. Run all cells.
### Option 2: Run locally
```bash
git clone https://github.com/<your-username>/stroke-prediction.git
cd stroke-prediction
pip install -r requirements.txt
jupyter notebook
```
Then update the `path` variable in the first code cell to point to your local copy of the CSV.
 
---
 
## Project Structure
 
```
stroke-prediction/
├── Stroke_Prediction.ipynb   # Full pipeline: EDA → ML/DL modelling → evaluation
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── models/
    ├── dl_model.keras        # Trained neural network
    └── scaler.pkl            # Fitted StandardScaler
```
 
---
 
## Limitations & Future Work
 
- **Handle imbalance during training, not only at prediction time.** Use `class_weight='balanced'` (scikit-learn) or `class_weight={0: 1, 1: 19}` (Keras), or resample the training set with SMOTE. This usually gives better precision at the same recall than extreme threshold tuning.
- **Use imbalance-aware metrics.** Report **PR-AUC**, ROC-AUC and the Matthews correlation coefficient (MCC is already imported), and choose the threshold from a precision–recall curve on a validation set.
- **Fix the engineered features.** Clinical thresholds (`glucose ≥ 126`, `bmi ≥ 30`, `age ≥ 60`) are applied after the power transform, when values are on a standardised scale. As a result these conditions are never true, so `elderly_smoker` and `married_senior` are constant (all zeros) and the risk scores lose their glucose, BMI and age components. Computing these features on raw values would restore them.
- **Code fix in the threshold experiment.** The cell that refits a Decision Tree then calls `model.predict_proba`, where `model` still points to the last model in the loop (XGBoost). The tuned-threshold result is therefore XGBoost's, and is reported as such above.
- **One-hot encode** `work_type` and `smoking_status`, since their integer codes imply an order that doesn't exist.
- **Prevent leakage.** The BMI median and the power transform are computed on the full dataset. Fitting them inside a scikit-learn `Pipeline` on the training data only, and saving the `PowerTransformer` alongside the scaler, would make inference reproducible.
- **Validation strategy:** early stopping monitors the test set. A separate validation split, or stratified k-fold cross-validation, would give an unbiased estimate.
---
 
## Disclaimer
 
This project is for **educational and research purposes only**. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.
 
---
 
## Author
 
**Patricia Fuglo**
 
⭐ If you found this project useful, consider giving it a star!
 
