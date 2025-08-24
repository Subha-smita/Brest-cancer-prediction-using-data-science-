import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# -------------------------------
# 1. Load and prepare data
# -------------------------------
data = pd.read_csv("data.csv")

# Drop useless columns
data = data.drop(["id", "Unnamed: 32"], axis=1)

# Target column
TARGET = "diagnosis"

X = data.drop(TARGET, axis=1)
y = data[TARGET]

# -------------------------------
# 2. Train models
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
log_reg = LogisticRegression(max_iter=5000).fit(X_train, y_train)
rf_model = RandomForestClassifier().fit(X_train, y_train)
svm_model = SVC(probability=True).fit(X_train, y_train)

# Compute accuracies
acc_log = log_reg.score(X_test, y_test)
acc_rf = rf_model.score(X_test, y_test)
acc_svm = svm_model.score(X_test, y_test)

# Best model
model_accuracies = {
    "Logistic Regression": acc_log,
    "Random Forest": acc_rf,
    "SVM": acc_svm
}
best_model_name = max(model_accuracies, key=model_accuracies.get)

# Helper function to decode results
def decode_prediction(pred):
    return "⚠️ Malignant (Breast Cancer Detected)" if pred == "M" else "✅ Benign (No Cancer)"

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.title("🔬 Breast Cancer Prediction App")
st.write("Provide the input features below to predict whether the tumor is **Benign** or **Malignant**.")

user_input = []
for feature in X.columns:
    val = st.number_input(
        f"{feature}",
        float(data[feature].min()),
        float(data[feature].max()),
        float(data[feature].mean())
    )
    user_input.append(val)

user_input = np.array(user_input).reshape(1, -1)
user_input = scaler.transform(user_input)

# -------------------------------
# 4. Predictions
# -------------------------------
if st.button("Predict"):
    pred_log = log_reg.predict(user_input)[0]
    pred_rf = rf_model.predict(user_input)[0]
    pred_svm = svm_model.predict(user_input)[0]

    st.subheader("🔎 Predictions from all models:")
    st.write(f"**Logistic Regression** → {decode_prediction(pred_log)}  (Accuracy: {acc_log:.2f})")
    st.write(f"**Random Forest** → {decode_prediction(pred_rf)}  (Accuracy: {acc_rf:.2f})")
    st.write(f"**SVM** → {decode_prediction(pred_svm)}  (Accuracy: {acc_svm:.2f})")

    # Best model result
    if best_model_name == "Logistic Regression":
        best_pred = pred_log
    elif best_model_name == "Random Forest":
        best_pred = pred_rf
    else:
        best_pred = pred_svm

    st.success(f"✅ Best Model: **{best_model_name}** → {decode_prediction(best_pred)} (Accuracy: {model_accuracies[best_model_name]:.2f})")
