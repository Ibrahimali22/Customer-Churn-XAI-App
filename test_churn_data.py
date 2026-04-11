import pandas as pd
import pytest
import os

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

@pytest.fixture(scope="module")
def load_data():
    """Loads the dataset once for all tests."""
    df = pd.read_csv(DATA_URL)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

def test_no_negative_tenure(load_data):
    """Business Validation: Customer tenure cannot be logically negative."""
    df = load_data
    assert (df['tenure'] < 0).sum() == 0, "Found customers with negative tenure!"

def test_no_negative_charges(load_data):
    """Business Validation: Monthly charges must be positive."""
    df = load_data
    assert (df['MonthlyCharges'] < 0).sum() == 0, "Found negative monthly charges!"

def test_churn_values_binary(load_data):
    """Data Integrity: Churn should only contain 'Yes' or 'No'."""
    df = load_data
    valid_values = {'Yes', 'No'}
    assert set(df['Churn'].unique()).issubset(valid_values), "Churn column contains invalid categorical values!"
