# Student Dropout Prediction

## Project Overview

This project uses Machine Learning to predict whether a university student may be at risk of dropping out. The project uses Logistic Regression as a straightforward binary classification model to ensure accuracy and interpretability.

## Problem Statement

Student dropout can negatively affect students and educational institutions. The purpose of this project is to develop a predictive system that can identify potential dropout risks early using student-related academic and lifestyle information.

## Dataset

The dataset (`student_dropout_dataset_v3.csv`) contains 10,000 student records and 19 columns originally.
It includes 17 input features covering academic performance (GPA, attendance, assignment delays), demographics, socioeconomic status (family income), and lifestyle factors (stress index, part-time jobs).

## Target Variable

The target variable is `Dropout`:
- 0 = Stayed (Enrolled)
- 1 = Dropped Out

## Preprocessing

The following data preparation steps were performed:
- Missing value analysis
- Missing value handling (Median imputation for numerical features, Mode for categorical)
- Duplicate checking
- Removal of the `Student_ID` column
- Ordinal Encoding (used to maintain the exact 17-feature structure)
- Feature Scaling using `StandardScaler`
- Train/Test Split (80% training / 20% testing)

## Exploratory Data Analysis (EDA)

EDA was performed using visualizations to find key patterns:
- Dropout class distribution
- Academic impact boxplots (GPA, Attendance, Study hours, Assignment delays)
- Correlation analysis (Heatmap)
- Lifestyle impact countplots (Internet Access, Part-Time Job status)

## Model

- **Algorithm:** `LogisticRegression`
- **Purpose:** Binary classification to predict dropout likelihood based on mathematical relationships in the 17 features.

## Evaluation

The model was evaluated using:
- Confusion Matrix
- Classification Report
- Core Metrics: Accuracy, Precision, Recall, and F1-score
- ROC-AUC Score (measuring class separation ability)

## Student Risk Prediction

The deployed application accepts the 17 student data inputs and generates:
- **Dropout Probability:** A decimal value between 0 and 1.
- **Risk Level:**
  - Below 0.40 = Low Risk
  - 0.40 to 0.70 = Medium Risk
  - Above 0.70 = High Risk

## Educational Early-Warning Application

This project serves as a prototype early-warning system. University advisors and educational staff can use these predictions to proactively identify students who may benefit from support such as counseling, tutoring, mentoring, or financial assistance.

*Note: The prediction should not be treated as a final judgment about a student, but rather as an additional tool for academic support.*

## Technologies

- Python
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Gradio (Web Interface)
- Joblib (Model Saving)
- Jupyter Notebook / Google Colab

## Project Workflow

Dataset Loading → Missing Value Cleaning → Exploratory Data Analysis (EDA) → Preprocessing & Ordinal Encoding → Train/Test Split → Logistic Regression Training → Evaluation Metrics → Gradio Web App Deployment

## Author

Student Dropout Prediction — Machine Learning Internship Project
