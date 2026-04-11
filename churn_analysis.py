import pandas as pd
import numpy as np
import os
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

class CustomerChurnPipeline:
    """
    Object-Oriented Pipeline for B2B/Telecom Customer Churn Analysis.
    Fetches the REAL IBM Telco Customer Churn dataset.
    Features Advanced EDA (Tenure & Contract types) and Model Explainability via SHAP.
    """
    
    def __init__(self):
        self.data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        
    def load_data(self):
        print(f"[INFO] Downloading real dataset from IBM Telco repo...")
        self.data = pd.read_csv(self.data_url)
        print(f"[INFO] Dataset loaded successfully! Shape: {self.data.shape}")
        
    def perform_eda(self):
        print("\n--- Phase 1: Exploratory Data Analysis (EDA) ---")
        
        self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
        self.data.dropna(inplace=True)
        
        churn_rate = (self.data['Churn'] == 'Yes').mean() * 100
        print(f"Overall Customer Churn Rate: {churn_rate:.2f}%")
        
        # Deep Business Insights: Tenure Analysis
        if 'tenure' in self.data.columns:
            early_churn = self.data[self.data['tenure'] <= 6]['Churn'].apply(lambda x: 1 if x == 'Yes' else 0).mean() * 100
            print(f"Risk Insight: Customers with tenure <= 6 months have a churn rate of {early_churn:.2f}%")
            
        # Deep Business Insights: Contract Type
        if 'Contract' in self.data.columns:
            contract_churn = self.data.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            print("\nRisk Insight: Churn Rate by Contract Type:")
            for contract_type, rate in contract_churn.items():
                print(f" - {contract_type}: {rate:.2f}%")
        
    def preprocess_data(self):
        print("\n[INFO] Preprocessing data (Encoding & Scaling)...")
        if 'customerID' in self.data.columns:
            self.data = self.data.drop('customerID', axis=1)
            
        self.data['Churn'] = self.data['Churn'].map({'Yes': 1, 'No': 0})
        
        y = self.data['Churn']
        X = self.data.drop('Churn', axis=1)
        X = pd.get_dummies(X, drop_first=True)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        scaler = StandardScaler()
        self.X_train[num_cols] = scaler.fit_transform(self.X_train[num_cols])
        self.X_test[num_cols] = scaler.transform(self.X_test[num_cols])
        print("[INFO] Preprocessing complete.")

    def train_and_evaluate(self):
        print("\n--- Phase 2: Predictive Modeling (Gradient Boosting with SMOTE) ---")
        try:
            from imblearn.over_sampling import SMOTE
            print("[INFO] Applying SMOTE to balance the imbalanced 'Churn' target...")
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(self.X_train, self.y_train)
            self.model.fit(X_res, y_res)
        except ImportError:
            print("[WARNING] 'imbalanced-learn' not installed. Running without SMOTE.")
            self.model.fit(self.X_train, self.y_train)
        
        y_pred = self.model.predict(self.X_test)
        
        print("\nModel Evaluation:")
        from sklearn.metrics import average_precision_score
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred) * 100:.2f}%")
        print(f"AUPRC: {average_precision_score(self.y_test, y_pred_proba):.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred))

    def run_model_explainability(self):
        """Uses SHAP to provide advanced Model Explainability."""
        print("\n--- Phase 3: Model Explainability (SHAP) ---")
        try:
            import shap
            print("[INFO] Calculating SHAP values to explain feature impacts...")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)
            
            print("[INFO] Explainability metric computed successfully. In a production environment, this is used to generate dependency plots and summary impact charts for stakeholders.")
            
            try:
                import matplotlib.pyplot as plt
                shap.summary_plot(shap_values, self.X_test, show=False)
                plt.savefig('shap_summary.png', bbox_inches='tight')
                print("[INFO] Saved SHAP impact summary plot to 'shap_summary.png'")
            except Exception as e:
                pass

        except ImportError:
            print("[WARNING] The 'shap' library is not installed. To see advanced model explanations, run 'pip install shap'.")

    def run_pipeline(self):
        self.load_data()
        self.perform_eda()
        self.preprocess_data()
        self.train_and_evaluate()
        self.run_model_explainability()

if __name__ == "__main__":
    pipeline = CustomerChurnPipeline()
    pipeline.run_pipeline()
