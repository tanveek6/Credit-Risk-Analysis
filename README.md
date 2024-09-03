# Credit Risk Analysis Using Machine Learning

## Overview
This project involves the development of a credit risk analysis system using machine learning algorithms.
The goal is to predict the likelihood of loan default based on various borrower characteristics. 
The project covers the entire machine learning pipeline, from data preprocessing and feature engineering to model development, evaluation, and deployment.

## Project Structure
- `credit_risk_analysis.ipynb`: The main Jupyter notebook where the entire analysis is performed.
- `credit_risk_model.pkl`: The trained Random Forest model saved for deployment.

## Dataset
The dataset used for this analysis contains various features about the borrower, loan details, and historical credit information.
The target variable is `loan_status`, which indicates whether the loan was defaulted.

### Features:
- `person_age`: Age of the borrower.
- `person_income`: Annual income of the borrower.
- `person_emp_length`: Length of employment in years.
- `person_home_ownership`: Type of home ownership.
- `loan_intent`: Purpose of the loan.
- `loan_amnt`: Loan amount.
- `loan_int_rate`: Interest rate of the loan.
- `loan_percent_income`: Percentage of income paid as loan.
- `cb_person_cred_hist_length`: Length of credit history.
- `cb_person_default_on_file`: Indicator of previous default history.

## Analysis Workflow

### 1. Data Preprocessing
- Handling Missing Values: Used `SimpleImputer` to fill missing values.
- Feature Scaling: Applied `StandardScaler` to standardize numerical features.
- Categorical Encoding: Used `OneHotEncoder` to encode categorical variables.

### 2. Addressing Class Imbalance
- SMOTE: Employed SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes in the training dataset.

### 3. Model Development
- Logistic Regression: A basic linear model to predict loan default.
- Random Forest Classifier: A more robust ensemble model that captures complex relationships.

### 4. Model Evaluation
- AUC-ROC Score: Evaluated models based on the Area Under the Receiver Operating Characteristic curve.
- Classification Report: Generated precision, recall, and F1-scores.

### 5. Risk Scoring
- Converted predictions into risk categories (Low, Medium, High) based on the Probability of Default (PD).

### 6. Model Validation
- Cross-Validation: Performed 5-fold cross-validation to validate the model's robustness.

### 7. Model Deployment
- Saved the Random Forest model using `joblib` for later deployment.
- Provided a Flask API outline for real-time predictions.

## How to Run

### Local Environment
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/credit-risk-analysis.git
    ```
2. Run the Jupyter notebook `credit_risk_analysis.ipynb` to see the entire analysis.

### Google Colab
1. Open `credit_risk_analysis.ipynb` in Google Colab.
2. Upload the dataset and run the cells sequentially.


