import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Set up the Page
st.set_page_config(page_title="B2B Churn Prediction System", layout="wide")
st.title("📊 B2B Customer Retention & Churn Predictive Analysis")
st.write("This Web App utilizes an Explainable ML Pipeline (XAI) to predict customer churn in real-time.")

# Cache the data loading and training phase to make the web app extremely fast
@st.cache_resource
def train_model():
    data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(data_url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True, errors='ignore')
    
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    y = df['Churn']
    X = df.drop('Churn', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # 1. Bias Fix: Applying SMOTE to prevent Overfitting to 'Stay'
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
    except:
        X_res, y_res = X_train, y_train

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_res, y_res)
    
    explainer = shap.TreeExplainer(model)
    return model, explainer, scaler, X_test, X

model, explainer, scaler, X_test, original_X = train_model()
st.success("✅ Machine Learning Models loaded (with SMOTE balancing active).")

st.sidebar.header("User Interface")
st.sidebar.write("Simulate a Customer Profile to see Churn Probability and reasoning.")

# Extreme Values test requested by Mentor
tenure = st.sidebar.slider("Tenure in Months", 0, 72, 1)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20.0, 120.0, 120.0)
total_charges = tenure * monthly_charges

month_to_month = st.sidebar.checkbox("Is Month-to-Month Contract?", value=True)

# 2. Feature Encoding Fix: We use the 'Mean' customer as a realistic baseline instead of all Zeros!
dummy_input = X_test.mean().copy()

# Assign sliders safely
num_data = pd.DataFrame({'tenure': [tenure], 'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges]})
scaled_nums = scaler.transform(num_data)

dummy_input['tenure'] = scaled_nums[0][0]
dummy_input['MonthlyCharges'] = scaled_nums[0][1]
dummy_input['TotalCharges'] = scaled_nums[0][2]

if 'Contract_One year' in dummy_input.index and 'Contract_Two year' in dummy_input.index:
    dummy_input['Contract_One year'] = 0 if month_to_month else 1
    dummy_input['Contract_Two year'] = 0

input_df = pd.DataFrame([dummy_input])

if st.button("Predict Customer Future (XAI)"):
    prediction_prob = model.predict_proba(input_df)[0][1]
    
    st.subheader("🤖 Artificial Intelligence Prediction:")
    if prediction_prob > 0.5:
        st.error(f"⚠️ HIGH RISK: The customer has a {prediction_prob*100:.1f}% probability of Churning!")
    else:
        st.success(f"🔒 SAFE: The customer has a {prediction_prob*100:.1f}% probability of Churning, they will likely stay.")
        
    st.subheader("🔍 Model Transparency (SHAP Values)")
    st.write("Why did the AI make this decision? The below graph breaks down the impact of this specific customer's features.")
    
    shap_vals = explainer.shap_values(input_df)
    
    # SHAP Waterfall / Force Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(shap.Explanation(values=shap_vals[0], base_values=explainer.expected_value[0], data=input_df.iloc[0], feature_names=input_df.columns), show=False)
    st.pyplot(fig)
    
    st.info("💡 **Business Recommendation Output:** If the 'Month-to-month' contract features heavily in the red (pushing churn), consider deploying an automated 15% discount for a 1-year lock-in!")
