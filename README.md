# DA25M001: DSA Assignment 3 - Addressing Class Imbalance in Fraud Detection

## Project Overview

This project tackles the critical problem of class imbalance in the context of credit card fraud detection. Using a standard credit card transaction dataset, we demonstrate how a typical classification model is biased towards the majority (non-fraudulent) class and fails to effectively identify fraudulent transactions.

The core objective is to implement and compare several resampling techniques—both oversampling and undersampling—to create a balanced training dataset and build a more effective fraud detection model. The performance of each technique is rigorously evaluated to provide a final, data-driven recommendation.

### Key Technologies
- Python 3
- Pandas & NumPy for data manipulation
- Scikit-learn for modeling (Logistic Regression, KMeans, StandardScaler)
- Imbalanced-learn for SMOTE
- Matplotlib for visualization
- KaggleHub for dataset access

---

## Dataset

The project utilizes the **Credit Card Fraud Detection** dataset, sourced from Kaggle.

- **Source:** [ULB ML Group on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Content:** The dataset contains transactions made by European cardholders. Features `V1` through `V28` are the result of a PCA transformation to protect user confidentiality. The only features that have not been transformed are `Time` and `Amount`. The target variable is `Class`, where `1` indicates a fraudulent transaction and `0` indicates a legitimate one.

### Class Distribution
The dataset is highly imbalanced, which is the central challenge of this project:
- **Normal Transactions (Class 0):** 284,315 (99.827%)
- **Fraudulent Transactions (Class 1):** 492 (0.173%)
- **Imbalance Ratio:** Approximately 578:1

This severe imbalance means a naive model can achieve over 99.8% accuracy by simply predicting "Normal" for every transaction, making it completely useless for detecting fraud.

---

## Methodology

The notebook is structured into three main parts, following a systematic approach to model building and evaluation.

### 1. Baseline Model
First, a baseline is established to measure the effectiveness of our resampling strategies.
- The data is split into training (80%) and testing (20%) sets, using stratification to maintain the original class distribution in both.
- Features are standardized using `StandardScaler`.
- A **Logistic Regression** model is trained on the original, imbalanced training data.
- Performance is evaluated on the imbalanced test set, focusing on **Precision, Recall, and F1-Score** for the minority (fraud) class.

### 2. Resampling Techniques
Three distinct techniques are implemented to balance the training data.

#### a) SMOTE (Synthetic Minority Over-sampling Technique)
- A naive oversampling method that creates new, synthetic minority samples by interpolating between existing minority instances in the feature space. This increases the number of minority samples to match the majority class.

#### b) CBO (Clustering-Based Oversampling)
- A more intelligent oversampling method designed to preserve the sub-structure of the minority class.
- The minority (fraud) samples are first clustered using **K-Means**. The optimal number of clusters (`k=5`) is determined using the Elbow Method.
- Synthetic samples are then generated *within each cluster*, ensuring that different types of fraudulent patterns are represented and reducing the generation of noisy samples.

#### c) CBU (Clustering-Based Undersampling)
- An intelligent undersampling method focused on reducing the majority class size while retaining its diversity.
- The majority (non-fraud) samples are clustered using **K-Means** (`k=10`).
- A representative subset of the majority class is created by proportionally selecting samples from each cluster, reducing its total size to match that of the original minority class.

### 3. Model Comparison and Analysis
- New Logistic Regression models are trained on each of the three balanced datasets (SMOTE, CBO, CBU).
- **Crucially, all four models (Baseline + 3 resampled) are evaluated on the same original, imbalanced test set.** This ensures a fair comparison of how well each model generalizes to real-world conditions.
- The results are compiled into a comparison table and visualized in bar charts.

---

## Results & Analysis

The performance of the four models on the imbalanced test set is summarized below:

| Model      | Precision | Recall  | F1-Score |
|------------|-----------|---------|----------|
| Baseline   | 0.8267    | 0.6327  | 0.7168   |
| SMOTE      | 0.0578    | 0.9184  | 0.1088   |
| CBO        | 0.0591    | 0.9184  | 0.1110   |
| CBU        | 0.0483    | 0.9184  | 0.0917   |

*(For a visual representation of these results, please see the bar charts in the Jupyter Notebook.)*

### Key Findings:
1.  **The Precision-Recall Trade-off:** The baseline model suffers from low recall, failing to detect over a third of fraudulent transactions. All resampling techniques successfully increased recall to ~92%, but at the expense of a catastrophic drop in precision to ~5%.
2.  **F1-Score Paradox:** Due to the collapse in precision, the baseline model technically has the highest F1-Score. However, it is not a viable solution because its low recall fails the primary goal of fraud detection.
3.  **CBO Superiority:** Among the resampled techniques, CBO achieved the highest precision and F1-score, suggesting that its clustering-based approach created a higher-quality training set compared to SMOTE and CBU for this problem.

---

## Conclusion & Recommendation

The recommended strategy for the company is to implement the **CBO-trained model as a high-sensitivity first-pass filter**.

This model excels at maximizing fraud detection (**91.8% Recall**), ensuring very few fraudulent cases go unnoticed. However, due to its low precision, it should not be used to automatically block transactions. Instead, transactions it flags as suspicious should be sent to a **second stage of verification** (e.g., a human review team or a secondary, high-precision model) to manage the high rate of false positives and ensure a positive customer experience.

---

.
├── DA5401_A3.ipynb       # The main Jupyter Notebook with all the code and analysis
└── README.md             # This file
```
