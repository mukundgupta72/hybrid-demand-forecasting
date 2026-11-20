from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_non_parametric_ensembles(X_train, y_train):
    print("[*] Training Non-Parametric Architectures (Random Forest & Gradient Boosting)...")
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    
    return rf_model, gb_model