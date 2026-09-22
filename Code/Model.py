import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('/content/student_dropout_dataset_v3.csv');
print("Shape:", df.shape)

for col in df:
  print(col)

df.dtypes

df["Dropout"].value_counts()

df.isnull().sum()

# Check for missing data
student_df=df
null_counts = student_df.isnull().sum()
null_counts = null_counts[null_counts > 0]

print("Columns with missing data:\n", null_counts)


total_duplicates = student_df.duplicated().sum()
print("Total duplicate rows found: ", total_duplicates)


num_missing_columns = ["Family_Income", "Study_Hours_per_Day", "Stress_Index"]

for col in num_missing_columns:
  student_df[col] = student_df[col].fillna(student_df[col].median())

student_df["Parental_Education"] = student_df["Parental_Education"].fillna(student_df["Parental_Education"].mode()[0])

print("\nMissing values remaining after imputation:\n", student_df.isnull().sum())

#Drop Student_ID column
student_df = student_df.drop(columns=['Student_ID'], errors='ignore')


X_features = student_df.drop("Dropout", axis=1)
y_target = student_df["Dropout"]

print("Features Shape:", X_features.shape)
print("Target Shape:", y_target.shape)


from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd

num_cols = X_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_features.select_dtypes(include=['object']).columns.tolist()

print("Numerical Columns:\n", num_cols)
print("\nCategorical Columns:\n", cat_cols)

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])

data_preprocessor = ColumnTransformer(
    transformers=[
        ('num_transform', num_pipeline, num_cols),
        ('cat_transform', cat_pipeline, cat_cols)
    ]
)

X_transformed = data_preprocessor.fit_transform(X_features)

print("\nShape before processing:", X_features.shape)
print("Shape after processing:", X_transformed.shape)




import matplotlib.pyplot as plt
import seaborn as sns


dropout_counts = df['Dropout'].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(['Stayed (0)', 'Dropped (1)'], dropout_counts.values, color=['#4C72B0', '#DD8452'])

plt.title('Distribution of Student Dropouts')
plt.xlabel('Dropout Status')
plt.ylabel('Student Count')
plt.show()

print("Percentage Breakdown:")
print(df["Dropout"].value_counts(normalize=True) * 100)




plt.figure(figsize=(7, 5))
sns.boxplot(x='Dropout', y='GPA', data=df)
plt.title('GPA vs Dropout Status')
plt.xlabel('Dropout (0 = No, 1 = Yes)')
plt.ylabel('GPA')
plt.show()


plt.figure(figsize=(7, 5))
sns.boxplot(x='Dropout', y='Attendance_Rate', data=df)
plt.title('Attendance Rate vs Dropout Status')
plt.xlabel('Dropout (0 = No, 1 = Yes)')
plt.ylabel('Attendance Rate (%)')
plt.show()


plt.figure(figsize=(7, 5))
sns.boxplot(x='Dropout', y='Study_Hours_per_Day', data=df)
plt.title('Study Hours vs Dropout Status')
plt.xlabel('Dropout (0 = No, 1 = Yes)')
plt.ylabel('Study Hours per Day')
plt.show()


plt.figure(figsize=(7, 5))
sns.boxplot(x='Dropout', y='Assignment_Delay_Days', data=df)
plt.title('Assignment Delay vs Dropout Status')
plt.xlabel('Dropout (0 = No, 1 = Yes)')
plt.ylabel('Assignment Delay (Days)')
plt.show()





numeric_features = [
    'Age', 'Family_Income', 'Study_Hours_per_Day', 'Attendance_Rate',
    'Assignment_Delay_Days', 'Travel_Time_Minutes', 'Stress_Index',
    'GPA', 'Semester_GPA', 'CGPA'
]

print("\nMean values grouped by Dropout status:")
print(df.groupby('Dropout')[numeric_features].mean().T)



plt.figure(figsize=(12, 8))
correlation_matrix = df[numeric_features + ['Dropout']].corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='Blues',  
    fmt='.2f'
)
plt.title('Correlation Matrix of Numerical Features')
plt.show()



plt.figure(figsize=(7, 5))
sns.countplot(x='Part_Time_Job', hue='Dropout', data=df)
plt.title('Part-Time Job vs Dropout Status')
plt.xlabel('Part-Time Job')
plt.ylabel('Student Count')
plt.show()



plt.figure(figsize=(7, 5))
sns.countplot(x='Internet_Access', hue='Dropout', data=df)
plt.title('Internet Access vs Dropout Status')
plt.xlabel('Internet Access')
plt.ylabel('Student Count')
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd


#Training/Testing

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_target, test_size=0.2, random_state=42)

print("Data Split Complete:")
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)



print("\nTraining the Machine Learning Model...")

dropout_model = LogisticRegression(random_state=42, max_iter=1000)

dropout_model.fit(X_train, y_train)

print("Model training successful!")



y_predictions = dropout_model.predict(X_test)

print("\nFirst 10 Model Predictions (0 = Stayed, 1 = Dropped):")
print(y_predictions[:10])


y_probabilities = dropout_model.predict_proba(X_test)

print("\nFirst 10 Prediction Probabilities:")
print("[Probability of Staying (0), Probability of Dropping (1)]")

for i in range(10):
    prob_stay = round(y_probabilities[i][0] * 100, 2)
    prob_drop = round(y_probabilities[i][1] * 100, 2)
    print(f"Student {i+1}: {prob_stay}% Stay | {prob_drop}% Drop")


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

cm = confusion_matrix(y_test, y_predictions)
print("Confusion Matrix:")
print(cm)


cr = classification_report(y_test, y_predictions)
print("\nClassification Report:")
print(cr)

print("\nAccuracy:", accuracy_score(y_test, y_predictions)*100)
print("Precision:", precision_score(y_test, y_predictions))
print("Recall:", recall_score(y_test, y_predictions))
print("F1-score:", f1_score(y_test, y_predictions))

roc_auc = roc_auc_score(y_test, y_probabilities[:, 1])
print("ROC-AUC:", roc_auc)