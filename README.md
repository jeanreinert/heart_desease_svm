# Heart Disease Classification with SVM - Data Treatment

## Project Overview
This repository contains the second part of a machine learning assignment (IAA006 - Part 2) focused on improving the performance of a Support Vector Machine (SVM) model on a low-quality dataset through data preprocessing techniques.

The chosen dataset is the Heart Disease dataset from the UCI Machine Learning Repository. The objective was to classify instances as either healthy (0) or having heart disease (1). The initial baseline SVM model performed poorly, achieving a 68% accuracy. Through a series of minimal but effective data treatment steps—imputation, encoding, and normalization—the model's accuracy was significantly improved to over 90%.

## Dataset
*   **Source:** UCI Machine Learning Repository - Heart Disease (processed cleveland data)
*   **Link:** https://archive.ics.uci.edu/dataset/45/heart+disease
*   **Task:** Binary Classification (0: Healthy, 1: Heart Disease)
    *   *Note: Original classes 1-4 were collapsed into a single 'Heart Disease' class (1) to ensure balanced and reliable predictions.*

## Preprocessing Interventions
The project demonstrates the impact of proper data preparation on model performance. The following steps were implemented to enhance the dataset:

1.  **Imputation (Missing Values):**
    *   Identified 6 rows with missing (NaN) values for categorical variables (ca - number of major vessels, and thal - thalassemia).
    *   Instead of dropping these rows (which would reduce the already small dataset), missing values were filled using the Mode (Most Frequent) strategy via SimpleImputer.
2.  **Categorical Encoding:**
    *   Several variables were identified as categorical (cp, restecg, slope, thal).
    *   Applied One-Hot Encoding (pd.get_dummies with drop_first=True) to convert these categories into numerical format suitable for the SVM algorithm.
3.  **Data Normalization:**
    *   Numerical variables (age, chol, thalach, oldpeak) had vastly different scales (e.g., age max 77 vs. chol max 564).
    *   Applied Min-Max Scaling (Linear Interpolation) to normalize all features to a range of 0 to 1, preventing features with larger scales from dominating the SVM's distance calculations.

## Results and Conclusion
The model was evaluated using a Holdout split (75% training, 25% testing) with stratification.

| Metric | Baseline (Untreated) | Improved (Treated) |
| :--- | :--- | :--- |
| **Accuracy** | 68.00% | 90.79% |

### Performance Comparison
| Result Type | Baseline Model | Improved Model |
| :--- | :--- | :--- |
| **True Negatives** | 33 | 36 |
| **False Positives** | 7 | 5 |
| **False Negatives** | 17 | 2 |
| **True Positives** | 18 | 33 |

**Conclusion:** The application of Data Imputation, One-Hot Encoding, and Min-Max Scaling successfully increased the SVM model's accuracy from 68% to nearly 91%, well exceeding the assignment's improvement goals.

## Instructions to Run
1.  Open the Jupyter Notebook (Trabalho IAA006 - Parte 2.ipynb).
2.  Ensure you have pandas and scikit-learn installed in your environment.
3.  Run all cells sequentially to observe the data loading, baseline training, preprocessing steps, and the final improved model training.
