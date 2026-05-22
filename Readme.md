# 🧠 Parkinson's Disease Detection via Voice Analysis

> **ACNEI Computational Neuroscience Introductory School — May 2026**
> A machine learning pipeline that detects Parkinson's Disease and predicts symptom severity from non-invasive voice recordings.

---

## 👥 Team Members

| Name | Role |
|------|------|
| Nicholas Baffoe | - |
| - | - |
| - | - |

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Background](#-background)
3. [Problem Statement](#-problem-statement)
4. [Aim & Objectives](#-aim--objectives)
5. [Dataset](#-dataset)
6. [Method & Pipeline](#-method--pipeline)
7. [Models Used](#-models-used)
8. [Results — Classification](#-results--part-1-classification)
9. [Results — Telemonitoring](#-results--part-2-telemonitoring-severity-prediction)
10. [How to Explain the Charts](#-how-to-explain-the-charts)
11. [Expected Outcomes](#-expected-outcomes)
12. [Limitations](#-limitations)
13. [Future Work](#-future-work)
14. [How to Run the Code](#-how-to-run-the-code)
15. [File Structure](#-file-structure)
16. [References](#-references)

---

## 🔬 Project Overview

This project uses **machine learning and voice biomarkers** to tackle two problems:

- **Part 1 — Classification:** Can we tell from a voice recording whether someone has Parkinson's Disease?
- **Part 2 — Telemonitoring:** Can we predict how severe their symptoms are, and track changes over time?

We use the [UCI Parkinson's Voice Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons), a clean, well-documented dataset of 195 voice recordings from 31 people (23 with Parkinson's, 8 healthy). We also use the accompanying [Telemonitoring dataset](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring) with 5,875 recordings from 42 early-stage patients tracked over 6 months.

The entire pipeline — from data loading to charts and model evaluation — is written in a single Python script.

---

## 🧬 Background

Parkinson's Disease (PD) is a **progressive neurodegenerative disorder** that affects dopamine-producing neurons in the brain. It disrupts motor control, causing tremors, muscle rigidity, and — critically — noticeable changes in speech and voice that often appear **before** a formal diagnosis.

**Key facts:**
- Over **10 million** people worldwide are affected
- Around **30%** of cases are diagnosed too late, after significant neuronal damage
- Around **90%** of Parkinson's patients develop voice abnormalities early in the disease

These voice changes include:
- **Increased Jitter** — instability in the pitch of the voice
- **Increased Shimmer** — instability in the volume/amplitude of the voice
- **Higher Noise Ratios (NHR)** — more noise relative to pure tone in the voice
- **Reduced HNR** — lower harmonics-to-noise ratio, indicating a breathier voice
- **Nonlinear changes** — captured by measures like PPE, DFA, RPDE, and D2

Because these features can be extracted computationally from a simple voice recording, they make a powerful and accessible screening signal.

---

## ❗ Problem Statement

Current Parkinson's diagnosis relies on **subjective clinical observation** by a specialist. This approach has four key problems:

| Problem | Detail |
|---------|--------|
| 💰 **High cost** | Specialist neurology visits are expensive and often not covered by insurance in lower-income regions |
| ⏱️ **Late detection** | ~30% of cases diagnosed only after major neuronal damage has occurred |
| 📍 **Limited access** | Rural and low-income communities lack specialist neurologists |
| 📉 **No remote option** | Patients must visit a clinic even just to track if their condition is worsening |

**The gap:** There is no widely available, objective, non-invasive, and scalable tool for Parkinson's screening or remote monitoring.

**Our proposed solution:** Use voice recordings and machine learning to fill this gap — a method that is low-cost, non-invasive, and can be administered remotely from a patient's home.

---

## 🎯 Aim & Objectives

### Main Aim
> To develop and evaluate a machine learning pipeline that detects Parkinson's Disease and predicts symptom severity from non-invasive voice recordings, using the UCI Parkinson's Voice Dataset.

### Objectives

**Objective 01 — Explore and analyse the voice data**
Perform exploratory data analysis (EDA) to understand feature distributions, class balance, and correlations between voice features and Parkinson's status.

**Objective 02 — Build a classification model**
Train and compare multiple ML classifiers (Random Forest, Gradient Boosting, SVM, Logistic Regression, KNN) to distinguish Parkinson's patients from healthy individuals based on voice measurements.

**Objective 03 — Predict disease severity**
Use the telemonitoring dataset to build regression models that predict UPDRS severity scores from voice biomarkers, enabling remote tracking of disease progression over time.

---

## 📦 Dataset

### Part 1 — Classification Dataset (`parkinsons.data`)

| Detail | Info |
|--------|------|
| **Samples** | 195 voice recordings |
| **Patients** | 31 (23 Parkinson's, 8 Healthy) |
| **Features** | 22 voice measurements |
| **Target** | `status` — 1 = Parkinson's, 0 = Healthy |
| **Missing values** | None ✅ |
| **Class balance** | 147 Parkinson's vs 48 Healthy (3:1 ratio) |

### Part 2 — Telemonitoring Dataset (`telemonitoring/parkinsons_updrs.data`)

| Detail | Info |
|--------|------|
| **Samples** | 5,875 voice recordings |
| **Patients** | 42 early-stage Parkinson's patients |
| **Features** | 16 voice measures + age, sex, test_time |
| **Targets** | `motor_UPDRS` and `total_UPDRS` (severity scores) |
| **Trial period** | 6 months, recorded at home |
| **Missing values** | None ✅ |

### Key Feature Groups

| Group | Features | What they measure |
|-------|----------|-------------------|
| **Jitter** | MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP | Pitch/frequency instability |
| **Shimmer** | MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA | Amplitude/volume instability |
| **Noise ratios** | NHR, HNR | Ratio of noise to tonal components |
| **Nonlinear** | RPDE, D2, DFA, PPE, spread1, spread2 | Complexity and irregularity of voice signal |

---

## ⚙️ Method & Pipeline

The entire pipeline runs in one Python script: `parkinsons_detection.py`

```
parkinsons.data
       │
       ▼
  ┌─────────────────────────────────────────────────┐
  │  STEP 1: Load data & verify shape/missing values │
  └─────────────────────────┬───────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────┐
  │  STEP 2: EDA                                     │
  │  • Class balance bar chart                       │
  │  • Correlation heatmap (top 10 features)         │
  │  • Jitter & Shimmer distribution histograms      │
  │  • Boxplots — top 6 features by class            │
  └─────────────────────────┬───────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────┐
  │  STEP 3: Preprocessing                           │
  │  • 80/20 stratified train-test split             │
  │  • StandardScaler (z-score normalisation)        │
  └─────────────────────────┬───────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────┐
  │  STEP 4: Train 5 classifiers                     │
  │  • Random Forest, Gradient Boosting, SVM,        │
  │    Logistic Regression, KNN                      │
  │  • 5-Fold Stratified Cross-Validation            │
  └─────────────────────────┬───────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────┐
  │  STEP 5: Evaluate                                │
  │  • Confusion matrices (all 5 models)             │
  │  • ROC curves with AUC scores                    │
  │  • Model comparison bar chart (CV vs Test)       │
  │  • Classification report + Feature importance    │
  └─────────────────────────┬───────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────┐
  │  STEP 6: Telemonitoring (Part 2)                 │
  │  • Load parkinsons_updrs.data                    │
  │  • UPDRS EDA (distributions + progression)       │
  │  • Train 3 regression models                     │
  │  • Predicted vs Actual scatter plots             │
  │  • Feature importance for severity prediction    │
  └─────────────────────────────────────────────────┘
```

---

## 🤖 Models Used

### Classification (Part 1)

| Model | Why we used it |
|-------|---------------|
| **Random Forest** ⭐ | Handles correlated features well, robust to noise, provides feature importances |
| **Gradient Boosting** | Strong on tabular data; builds on errors of previous trees sequentially |
| **SVM (RBF kernel)** | Effective in high-dimensional spaces; maximises the decision margin |
| **Logistic Regression** | Interpretable linear baseline; good for probability estimates |
| **KNN (k=7)** | Non-parametric; captures local structure in the feature space |

> **Note:** Tree-based models (RF, GBM) were trained on raw features. Distance-based models (SVM, LR, KNN) used StandardScaler-normalised features.

### Regression (Part 2)

| Model | Why we used it |
|-------|---------------|
| **Random Forest** ⭐ | Non-linear, handles mixed feature types, gives feature importances |
| **Gradient Boosting** | Strong iterative learner for continuous targets |
| **Ridge Regression** | Regularised linear baseline — useful for comparison |

---

## 📊 Results — Part 1: Classification

### Cross-Validation vs Test Accuracy

| Model | CV Accuracy | ± Std | Test Accuracy |
|-------|-------------|-------|---------------|
| **Random Forest** ⭐ | **89.76%** | 0.0442 | **92.31%** |
| Gradient Boosting | 89.13% | 0.0576 | 92.31% |
| SVM (RBF) | 85.28% | 0.0514 | 92.31% |
| Logistic Regression | 82.08% | 0.0523 | 92.31% |
| K-Nearest Neighbours | 85.24% | 0.0489 | 87.18% |

### Best Model — Random Forest

```
Classification Report (Random Forest):

              precision    recall  f1-score   support

     Healthy       0.80      0.80      0.80        10
 Parkinson's       0.94      0.94      0.94        29

    accuracy                           0.92        39
   macro avg       0.87      0.87      0.87        39
weighted avg       0.91      0.92      0.91        39
```

### Top Voice Features (by importance)

1. **PPE** — Nonlinear measure of fundamental frequency variation
2. **spread1** — Nonlinear measure of fundamental frequency variation
3. **HNR** — Harmonics-to-noise ratio (lower = breathier voice)
4. **MDVP:Fo(Hz)** — Average vocal fundamental frequency
5. **DFA** — Signal fractal scaling exponent

---

## 📈 Results — Part 2: Telemonitoring Severity Prediction

### Regression Results

| Model | Motor UPDRS R² | Total UPDRS R² | Motor MAE | Total MAE |
|-------|---------------|----------------|-----------|-----------|
| **Random Forest** ⭐ | **0.335** | **0.361** | ~4.2 | ~5.1 |
| Gradient Boosting | 0.247 | 0.261 | ~4.8 | ~5.8 |
| Ridge Regression | 0.076 | 0.079 | ~6.1 | ~7.2 |

### Interpreting the R² Score

An R² of ~0.36 means our model explains about **36% of the variation** in UPDRS severity from voice alone. This is **moderate but meaningful**, and is consistent with published research on this dataset.

> UPDRS scores are influenced by many non-acoustic factors (medication, physical examination, cognitive state). Voice alone cannot fully predict severity — but it captures a real and useful signal, which is exactly what makes it valuable for remote monitoring.

---

## 📉 How to Explain the Charts

Use this section to prepare your presentation. Each chart is explained below.

---

### Chart 1 — Class Distribution Bar Chart (`eda_overview.png`, left)

**What it shows:** Two bars — 147 Parkinson's samples vs 48 Healthy samples.

**Interpretation:**
> *This bar chart shows our class balance. We have 147 Parkinson's recordings and 48 healthy ones: a 3 to 1 ratio. This imbalance reflects real-world clinic data, so we accounted for it using stratified train-test splitting to make sure both classes were fairly represented during training and testing.*

---

### Chart 2 — Feature Correlation Heatmap (`eda_overview.png`, right)

**What it shows:** A colour grid showing how the top 10 features relate to each other and to the target (Parkinson's status). Red = strong positive correlation, blue = negative.

**Interpretation:**
> *"This heatmap shows which voice features are most related to the Parkinson's diagnosis. The darker red squares near the status column tell us those features are the strongest predictors. We can also see some features are highly correlated with each other, which is why tree-based models like Random Forest handle this dataset well: they naturally select the most informative ones."*

---

### Chart 3 — Jitter & Shimmer Histograms (`feature_distributions.png`)

**What it shows:** 6 overlapping histograms — red for Parkinson's, blue for Healthy — for the top 3 Jitter and 3 Shimmer features.

**Interpretation:**
> *"These histograms compare how Jitter and Shimmer are distributed between the two groups. Jitter measures pitch instability and Shimmer measures volume instability in the voice. You can see the red Parkinson's bars spread further to the right, meaning their voices are significantly more unstable than healthy voices. This visual separation is why voice data is such a promising signal for detection."*

---

### Chart 4 — Boxplots (`boxplots.png`)

**What it shows:** 6 boxplots side by side for Healthy vs Parkinson's. The box = middle 50% of values, the centre line = median, dots = outliers.

**Interpretation:**
> *"These boxplots confirm what the histograms showed. For each of the top 6 features, the Parkinson's group has a noticeably higher median and wider spread than the healthy group. The bigger the gap between the two boxes, the more useful that feature is for our model. Features like PPE and spread1 show the clearest separation."*

---

### Chart 5 — Confusion Matrices (`confusion_matrices.png`)

**What it shows:** A 2×2 grid per model showing correct and incorrect predictions. Top-left = correctly predicted Healthy. Bottom-right = correctly predicted Parkinson's. The off-diagonal cells are errors.

**Interpretation:**
> *"A confusion matrix shows us where the model gets it right and where it makes mistakes. The diagonal from top-left to bottom-right are the correct predictions. For Random Forest, we can see it correctly identified 28 out of 29 Parkinson's patients and 8 out of 10 healthy ones. In a medical context, missing a Parkinson's patient is the most dangerous error — and our model keeps that number very low."*

---

### Chart 6 — ROC Curves (`roc_curves.png`)

**What it shows:** All 5 models on one chart. The closer the curve is to the top-left, the better. AUC (Area Under Curve) is shown in the legend — 1.0 is perfect, 0.5 is random.

**Interpretation:**
> *"The ROC curve measures how well each model separates the two classes across all possible decision thresholds. All our models sit well above the dashed random-baseline line. Random Forest achieves the highest AUC — meaning it has the best overall discrimination ability."*

---

### Chart 7 — Model Comparison Bar Chart (`model_comparison.png`)

**What it shows:** Side-by-side bars for CV accuracy (blue) vs test accuracy (red) for all 5 models.

**Interpretation:**
> *"This chart compares all five models side by side. The blue bar is the cross-validation accuracy — how well the model performed across 5 different train-test folds. The red bar is the final test accuracy on unseen data. The fact that these two bars are close together tells us our models are not memorising the training data:they generalise well. Random Forest came out on top."*

---

### Chart 8 — Feature Importance (`feature_importance.png`)

**What it shows:** Top 15 voice features ranked by how much they contributed to Random Forest's decisions. Longer bar = more important.

**Interpretation:**
> *"This chart tells us which voice features the Random Forest model relied on most. PPE: a nonlinear measure of pitch variation was the single most important feature, followed by spread1 and HNR, which measures the ratio of clean tone to noise in the voice. This makes biological sense: Parkinson's causes both tremor and breathiness in the voice, and these features capture exactly that"*

---

### Chart 9 — UPDRS EDA (`tele_eda.png`)

**What it shows:** Left panel — distribution of Motor and Total UPDRS severity scores. Right panel — average UPDRS scores over the 6-month trial period.

**Interpretation::**
> *"These two charts are from Part 2 — the telemonitoring dataset. On the left, we can see that most patients fall in the moderate severity range. On the right, we can track how the average scores creep upward over the 6-month trial: this confirms the progressive nature of the disease and shows why remote monitoring matters. Our regression model tries to predict these scores from voice alone."*

---

### Chart 10 — Predicted vs Actual UPDRS (`tele_predictions.png`)

**What it shows:** Scatter plots where each dot is one recording — x-axis is the real UPDRS score, y-axis is what the model predicted. The dashed line = perfect prediction.

**Interpretation:**
> *"This scatter plot compares what our model predicted versus what the actual UPDRS score was. If the model were perfect, every dot would land on the dashed line. We can see a clear upward trend: the model is definitely learning the pattern: but there is scatter around the line. An R² of 0.36 means voice features explain about 36% of the severity variation. That is meaningful, but it also honestly reflects that UPDRS is influenced by many factors beyond just voice."*

---

## ✅ Expected Outcomes

### Outcome 1 — High-accuracy Parkinson's classifier
We expect our best model (Random Forest) to achieve **above 90% classification accuracy**, correctly identifying Parkinson's patients from voice recordings.

**Why it matters:** This demonstrates that voice biomarkers are a viable, non-invasive screening tool — enabling early detection without specialist visits.

### Outcome 2 — Meaningful severity prediction
Our regression models are expected to achieve a **moderate R² score (0.30–0.40)** when predicting UPDRS severity from voice alone, consistent with published research.

**Why it matters:** Even partial prediction is clinically useful — it enables remote home monitoring and reduces clinic visits.

### Outcome 3 — Interpretable feature insights
Feature importance analysis is expected to confirm that **PPE, spread1, and HNR** are the strongest voice biomarkers.

**Why it matters:** This connects ML findings to neuroscience — these features reflect laryngeal motor impairment caused by Parkinson's, grounding our results in biology.

---

## ⚠️ Limitations

- **Small dataset:** Only 195 samples from 31 patients for the classification task
- **Class imbalance:** 3:1 ratio of Parkinson's to Healthy — the model sees far more Parkinson's examples
- **Voice only:** No imaging, clinical, or genetic data — voice alone cannot replace full clinical assessment
- **Single dataset source:** All recordings from one study — results may not generalise to all populations
- **Moderate regression R²:** UPDRS is influenced by non-acoustic factors our model cannot capture

---

## 🔭 Future Work

- Collect larger, more diverse datasets across different languages and demographics
- Apply deep learning models (CNNs, LSTMs) directly to raw audio waveforms
- Combine voice data with wearable sensor data for richer feature sets
- Build a mobile app that allows patients to record their voice and receive a risk score
- Test in clinical settings to validate real-world performance

---

## 🚀 How to Run the Code

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Folder structure

```
📁 your_project_folder/
├── parkinsons_detection.py     ← main script
├── parkinsons.data              ← classification dataset
├── README.md                    ← this file
└── telemonitoring/
    └── parkinsons_updrs.data    ← severity dataset
```

### Run

```bash
python parkinsons_detection.py
```

All 10 charts will be saved automatically into an `output_charts/` folder.

### What the script does

| Section | What runs |
|---------|-----------|
| Steps 1–3 | Load data, EDA, preprocessing |
| Steps 4–6 | Train 5 classifiers, cross-validate |
| Steps 7–9 | Confusion matrices, ROC curves, model comparison |
| Steps 10–11 | Best model report + feature importance |
| Part 2 (T1–T6) | Telemonitoring EDA, regression, predicted vs actual plots |

---

## 📁 File Structure

```
📁 project/
├── parkinsons_detection.py          ← Full ML pipeline (classification + telemonitoring)
├── parkinsons.data                  ← UCI classification dataset
├── README.md                        ← This file
├── Parkinsons_ACNEI_Proposal.pptx   ← 7-slide PowerPoint presentation
├── telemonitoring/
│   └── parkinsons_updrs.data        ← Telemonitoring dataset
└── output_charts/                   ← Auto-generated when you run the script
    ├── eda_overview.png             ← Class balance + correlation heatmap
    ├── feature_distributions.png   ← Jitter & Shimmer histograms
    ├── boxplots.png                 ← Top 6 features by class
    ├── confusion_matrices.png       ← All 5 models
    ├── roc_curves.png               ← ROC + AUC for all models
    ├── model_comparison.png         ← CV vs test accuracy
    ├── feature_importance.png       ← Top 15 features (Random Forest)
    ├── tele_eda.png                 ← UPDRS distributions + progression
    ├── tele_predictions.png         ← Predicted vs actual UPDRS
    └── tele_feature_importance.png  ← Top voice features for severity
```

---

## 📚 References

1. Little, M.A., McSharry, P.E., Roberts, S.J., Costello, D.A.E., Moroz, I.M. (2007). *Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection.* BioMedical Engineering OnLine.

2. Tsanas, A., Little, M.A., McSharry, P.E., Ramig, L.O. (2010). *Accurate telemonitoring of Parkinson's disease progression by noninvasive speech tests.* IEEE Transactions on Biomedical Engineering.

3. UCI Machine Learning Repository — [Parkinsons Data Set](https://archive.ics.uci.edu/ml/datasets/parkinsons)

4. UCI Machine Learning Repository — [Parkinsons Telemonitoring](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring)

---

## 📄 Licence

This project is for academic and educational purposes only.
Dataset credit: UCI Machine Learning Repository.

---

*Built for the ACNEI Computational Neuroscience Introductory School, May 2026.*
