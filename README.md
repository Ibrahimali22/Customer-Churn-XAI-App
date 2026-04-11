# B2B Customer Retention & Churn Predictive System

## Project Overview
Customer retention is crucial for Telecom & SaaS businesses. This project involves deep exploratory data analysis (EDA) and robust predictive machine learning to identify customers at high risk of churning, transitioning raw operational logs into actionable business insights.

## Deep Business Insights & EDA
During initial Data Mining on the IBM Telco dataset, two massive retention drivers were discovered:
1. **The Contract Trap:** Customers holding **"Month-to-month"** contracts accounted for overwhelmingly the highest churn bracket (approx. 42%+), while 1-year and 2-year contracts had negligible churn. This is a critical product strategy insight.
2. **Early Attrition:** Tenure profiling revealed that a substantial majority of churn occurs within the very first 6 months of the customer landing, emphasizing a need for a stronger onboarding process.

## Tech Stack
- **Data Engineering:** Pandas, NumPy, Scikit-Learn
- **Predictive Algorithms:** `GradientBoostingClassifier`
- **Model Explainability:** `SHAP` (SHapley Additive exPlanations)

## Model Explainability (SHAP)
Predicting *who* will churn is only half the battle down in the real world. You must answer *why* they are churning. This pipeline integrates the **SHAP** library, mapping the exact localized impact of every single business feature (like high Monthly Charges vs Tech Support access) on an individual customer's prediction. This allows targeted intervention by the marketing team.

## 🚀 Professional Software Development Architecture
To elevate this project from a Data Science script into a Production-Grade ML Application, we implemented:
1. **Interactive Deployment (Streamlit):** Deployed a frontend UI (`app.py`) allowing non-technical HR or Marketing Executives to simulate a customer profile and instantly retrieve the AI Prediction paired with a live SHAP explainer dashboard.
2. **Data Validation (Automated Testing):** Used `Pytest` to establish CI/CD data integrity constraints ensuring no negative tenure limits or invalid categorical mapping leaks into the ML Model. An automated GitHub Action seamlessly integrates to validate this upon pushing code.

## 💰 Business Recommendations (Actionable Impact)
By analyzing the SHAP Output and the gradient dependencies, our model shifts into generating *Problem-Solution-Impact* insights:
- **Insight:** Month-to-month contracting is the primary driver of the massive ~40% Churn rate, specifically targeting customers who hit their 5-month breaking point.
- **Strategic Recommendation:** We recommend deploying an automated **15% Discount Offer** marketed specifically to month-to-month customers entering their 5th month, strictly conditional on them transitioning to an Annual Retainer contract. This drastically leverages data to cut high-revenue hemorrhage dynamically.

## 🛠️ How to Run locally

### 1. Install Dependencies
Ensure you have Python installed, then run the following command in your terminal to install all required libraries:
```bash
pip install pandas numpy scikit-learn shap streamlit pytest imbalanced-learn
```

### 2. Run Automated Data Validation
To execute the Pytest suite verifying the business logic and data schema constraints:
```bash
pytest test_churn_data.py
```

### 3. Launch the Interactive Web Application (Streamlit)
To start the XAI (Explainable AI) dashboard and interact with the churn prediction model in your browser:
```bash
streamlit run app.py
```
