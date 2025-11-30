# 🩺 Machine Learning Internship - Healthcare Disease Prediction

**Project Name:** `Cognetix_DiseasePrediction`
**Internship Domain:** Machine Learning / Healthcare Analytics
**Organization:** @Cognetix Technology

---

## 🎯 Objective

The objective of this project was to build a robust machine learning model to predict the likelihood of a patient having **diabetes** based on clinical and lifestyle data (Pima Indians Dataset). The focus was on optimizing performance metrics critical for medical diagnosis, particularly **Recall** (minimizing missed cases of the disease).

## ⚙️ Functional Requirements & Key Steps

This project successfully fulfilled all requirements, including advanced data handling and final deployment:

1.  **Medical Data Preprocessing:** Crucially addressed the issue of **missing values** represented by zeros (`0`) in clinical columns (`Glucose`, `BMI`, `BloodPressure`, etc.) by replacing them with the column's **median** to maintain data integrity.
2.  **Feature Scaling:** Applied **StandardScaler** to all numerical features to normalize the data before modeling.
3.  **Model Training & Comparison:** Trained and evaluated both **Logistic Regression** and the superior **Random Forest Classifier**.
4.  **Comprehensive Evaluation:** Evaluated performance using **Accuracy, Precision, Recall, F1-Score**, and the **ROC AUC** score.
5.  **Key Insight Visualization:** Generated plots for the **Confusion Matrix** and the **Feature Importance Plot** (Random Forest) .
6.  **Deployment:** Saved the trained model (`disease_prediction_rf_model.joblib`) and scaler (`scaler.joblib`) and deployed a real-time prediction demo using **Streamlit**.

---

## 📊 Results and Analysis (Random Forest Classifier)

The **Random Forest Classifier** was selected as the final model due to its better performance in balancing Precision and Recall, crucial for medical applications.

### Model Performance (Random Forest)

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy Score** | **0.7792** | Overall correct predictions. |
| **Precision Score** | **0.7273** | Of all patients predicted diabetic, 72.73% were correct. |
| **Recall Score** | **0.5926** | **Crucial:** 59.26% of actual diabetic cases were correctly identified. |
| **F1-Score** | **0.6531** | The balanced score indicating robust overall performance. |
| **ROC AUC Score** | **0.8191** | Excellent ability to distinguish between diabetic and non-diabetic cases. |

---

## 🔑 Key Contributing Features

The **Feature Importance** plot confirms that the **Glucose** level, **BMI**, and **Age** are the most critical features, consistent with medical understanding of diabetes risk factors.

---

## 🚀 Model Deployment (Streamlit)

The final model is deployed via a simple web application for demonstrating real-time predictions.

**Execution Command:**
```bash
streamlit run app.py
```
## 🛠️ Technology Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, joblib
* **Deployment:** Streamlit
* **Dataset Source:** Pima Indians Diabetes Database (`diabetes.csv`)

---

## **Project Done By**

### **Hemavarni S**
