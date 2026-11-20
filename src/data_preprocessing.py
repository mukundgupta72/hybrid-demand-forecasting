import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath="data/Walmart.csv"):
    print(f"[*] Extracting raw source features from real dataset: {filepath}")
    
    # Read the physical Walmart CSV file
    df = pd.read_csv(filepath)
    
    # Parse dates explicitly (handling DD-MM-YYYY format safely)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Day_of_Week'] = df['Date'].dt.weekday
    
    # Define Matrix of Independent Features (X) and Dependent Target Vector (y)
    # Including 'Store' mimics your Multi-Location dark store framework perfectly!
    feature_cols = ['Store', 'Day_of_Week', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    X = df[feature_cols].values
    y = df['Weekly_Sales'].values
    
    # Split data: 80% Training for model fitting, 20% Testing for validation
    return train_test_split(X, y, test_size=0.2, random_state=42)