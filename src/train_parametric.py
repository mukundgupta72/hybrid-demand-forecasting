# src/train_parametric.py
from sklearn.linear_model import LinearRegression

def train_parametric_baseline(X_train, y_train):
    print("[*] Training Parametric Architecture Baseline (Linear Regression)...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model