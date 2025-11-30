import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
import joblib # For saving/deploying the model
import sys

# --- 1. Data Loading and Initial Cleaning ---
file_path = 'diabetes.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please download it and ensure it is in the directory.")
    sys.exit()

# Clinical columns where 0 indicates a missing value, not an actual measurement
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0s with NaN for proper imputation
df[cols_to_replace] = df[cols_to_replace].replace(0, np.NaN)

# Impute NaN values with the median of the respective column
for col in cols_to_replace:
    df[col] = df[col].fillna(df[col].median())
print("Missing 0 values (e.g., BMI=0, BP=0) imputed with the median.")

# Define features (X) and target (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# --- 2. Exploratory Data Analysis (EDA) ---

# A. Outcome Distribution Check (Class Imbalance)
plt.figure(figsize=(6, 5))
sns.countplot(x=y)
plt.title('Outcome Distribution (0: Non-Diabetic, 1: Diabetic)')
plt.savefig('eda_outcome_distribution.png')
print("EDA: Outcome distribution plot saved.")

# B. Relationship between Glucose and Outcome
plt.figure(figsize=(8, 6))
sns.boxplot(x='Outcome', y='Glucose', data=df)
plt.title('Glucose Levels by Disease Outcome')
plt.savefig('eda_glucose_vs_outcome.png')
# plt.show()
print("EDA: Glucose vs. Outcome box plot saved.")

# --- 3. Data Preprocessing and Splitting ---

# A. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# B. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

# --- 4. Model Training and Evaluation ---

def evaluate_model(model, X_test, y_test, model_name):
    """Trains, predicts, and evaluates a given model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability for ROC/AUC

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_proba)

    print(f"\n--- Model: {model_name} ---")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"Precision Score: {precision:.4f}")
    print(f"Recall Score: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC Score: {auc_score:.4f}")

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease (0)', 'Disease (1)'], yticklabels=['No Disease (0)', 'Disease (1)'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    
    # ROC Curve Visualization
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    
    return y_proba

# A. Logistic Regression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
evaluate_model(logreg, X_test, y_test, "Logistic Regression")

# B. Random Forest Classifier (Chosen for Feature Importance)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_proba = evaluate_model(rf_model, X_test, y_test, "Random Forest")


# --- 5. Feature Importance and Model Saving ---

# A. Feature Importance Plot (using Random Forest)
print("\n--- Feature Importance Visualization ---")
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Random Forest Feature Importance for Disease Prediction')
plt.xlabel('Feature Importance Score')
plt.ylabel('Health Indicator')
plt.tight_layout()
plt.savefig('feature_importance_disease.png')
print("Feature importance plot saved as 'feature_importance_disease.png'.")

# B. Save Model and Scaler for Deployment
joblib.dump(rf_model, 'disease_prediction_rf_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("\nModel and Scaler saved for deployment: 'disease_prediction_rf_model.joblib' and 'scaler.joblib'.")