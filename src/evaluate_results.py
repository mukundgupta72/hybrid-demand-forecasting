import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def execute_empirical_evaluation(predictions_dict, y_test):
    print("[*] Calculating statistical research matrices...")
    metrics_summary = {}

    for name, predictions in predictions_dict.items():
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        metrics_summary[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    print("\n" + "=" * 78)
    print("                  RESEARCH PERFORMANCE EVALUATION METRICS")
    print("=" * 78)
    print(f"{'Model Architecture':<36} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<8}")
    print("-" * 78)
    for name, m in metrics_summary.items():
        print(f"{name:<36} | {m['MAE']:<10.2f} | {m['RMSE']:<10.2f} | {m['R2']:<8.4f}")
    print("=" * 78 + "\n")

    short_names = [
        "Linear\nRegression",
        "Gradient\nBoosting",
        "Deep\nLSTM",
        "Random\nForest",
        "Hybrid\n(RF + LSTM)"
    ]
    r2_scores = [m["R2"] for m in metrics_summary.values()]
    rmse_scores = [m["RMSE"] for m in metrics_summary.values()]

    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    ax1.set_ylabel('R2 Score (Higher is Better)', color='navy', fontweight='bold')
    ax1.bar(short_names, r2_scores, color='lightblue', alpha=0.85, width=0.4)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor='navy')

    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE Value (Lower is Better)', color='crimson', fontweight='bold')
    ax2.plot(short_names, rmse_scores, color='crimson', marker='o', linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor='crimson')

    plt.title('Hybrid Deep Sequence & Ensemble Error Comparison', fontsize=12, fontweight='bold', pad=15)
    fig.tight_layout()

    plt.savefig("hybrid_comparison_chart.png", dpi=300)
    print("[+] Chart saved as 'hybrid_comparison_chart.png'")