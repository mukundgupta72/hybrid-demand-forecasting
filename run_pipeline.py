import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_preprocessing import load_and_preprocess_data
from src.train_parametric import train_parametric_baseline
from src.train_non_parametric import train_non_parametric_ensembles
from src.train_deep_sequence import train_and_predict_lstm
from src.evaluate_results import execute_empirical_evaluation

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    lr_model = train_parametric_baseline(X_train, y_train)
    rf_model, gb_model = train_non_parametric_ensembles(X_train, y_train)

    lr_preds = lr_model.predict(X_test)
    gb_preds = gb_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)
    lstm_preds = train_and_predict_lstm(X_train, y_train, X_test)

    hybrid_preds = (0.75 * rf_preds) + (0.25 * lstm_preds)

    all_predictions = {
        "Linear Regression (Parametric)": lr_preds,
        "Gradient Boosting (Ensemble)": gb_preds,
        "Deep LSTM (Sequence Network)": lstm_preds,
        "Random Forest (Ensemble)": rf_preds,
        "Hybrid Framework (RF + LSTM)": hybrid_preds
    }

    execute_empirical_evaluation(all_predictions, y_test)

if __name__ == "__main__":
    main()