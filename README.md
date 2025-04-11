# Customer Churn Prediction

This project builds a machine learning pipeline to predict customer churn using structured banking data. It covers data preprocessing, feature engineering, model training, evaluation, and performance comparison across multiple algorithms.

---

## Dataset

The dataset used is `Churn_Modeling.csv`, which contains customer-level data from a bank, including:

- **Demographics**: Geography, Gender, Age
- **Account Info**: Credit Score, Balance, Tenure, Number of Products
- **Behavioral Features**: IsActiveMember, HasCrCard
- **Target Variable**: `Exited` (1 = Churned, 0 = Retained)

---

## Feature Engineering

The following additional features were created to boost model performance:

- `BalanceZero`: Binary flag for zero balance accounts
- `AgeGroup`: Age segmented into bins
- `BalanceToSalaryRatio`: Normalized spending power
- `ProductUsage`: Interaction between product count and activity
- `TenureGroup`: Tenure binned into categorical groups
- `Male_Germany`, `Male_Spain`: Gender × Geography interaction terms

Categorical features were encoded using `LabelEncoder` and `pd.get_dummies`.

---

##  Models Trained

Each model is trained on scaled features and evaluated using confusion matrix, accuracy, and classification metrics:

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Gradient Boosting Classifier

---

## Workflow

1. Load and clean data
2. Feature engineering
3. Encode categorical variables
4. Train-test split
5. Feature scaling (`StandardScaler`)
6. Train and evaluate multiple models

---

##  Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score (via `classification_report`)

---

## Tools Used

- Python 3.9+
- Jupyter Notebook
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

---



