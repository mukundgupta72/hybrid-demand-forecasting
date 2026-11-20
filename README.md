# Hybrid-Demand-Forecasting-Research

> A Hybrid Deep Sequence and Ensemble Learning Framework for Multi-Location Supply Chain Demand Forecasting

---

## Project Overview
- **The Goal:** To replace erratic, traditional supply chain guesswork with a unified deep sequence and ensemble machine learning framework.
- **The Core Discovery:** Parametric models (Linear Regression) fail under market volatility (14.8% accuracy), Deep LSTM networks decode non-linear sequential shifts (81.7% accuracy), and Non-Parametric tree ensembles alongside the Hybrid (RF + LSTM) architecture capture up to 93.3% of complex retail demand variance.
- **The Application:** A modular, production-ready optimization engine designed for microservices deployment to prevent multi-location dark store stockouts and minimize inventory holding costs.

---

## Setup & Execution Guide

Follow these steps to set up the local environment and execute the machine learning pipeline on your machine:

### 1. Clone or Initialize Your Directory
Ensure your folder layout matches the following structured repository blueprint:

```text
Hybrid-Demand-Forecasting-Research/
├── data/
│   └── Walmart.csv              # Multi-location retail & macroeconomic dataset
├── src/
│   ├── data_preprocessing.py    # Parses timestamps, extracts tabular & cyclical features
│   ├── train_parametric.py      # Parametric baseline execution engine (Linear Regression)
│   ├── train_non_parametric.py  # Tree ensemble engines (Random Forest & Gradient Boosting)
│   ├── train_deep_sequence.py   # Vectorized Deep LSTM sequence network with Adam optimizer
│   └── evaluate_results.py      # Evaluates statistical error matrices & generates plots
├── run_pipeline.py              # Unified pipeline execution controller
├── hybrid_comparison_chart.png  # Generated dual-axis R2 & RMSE comparison plot
├── requirements.txt             # Python package dependencies
└── README.md
```

### 2. Install Required Dependencies
Open your terminal inside the project root folder and execute the installation command:
```bash
pip install -r requirements.txt
```

### 3. Run the Research Pipeline
Execute the unified terminal command to trigger data preprocessing, multi-architecture training, statistical evaluation, and automated visualization generation:
```bash
python run_pipeline.py
```
---

## Core Research Summary

### 1. Problem Statement
Traditional retail supply chains rely on basic moving averages or rigid linear models that fail to capture overlapping demand drivers such as weekend shopping surges, promotional holiday events, regional temperature shifts, and macroeconomic fluctuations (Unemployment and CPI indices). This research isolates the exact empirical boundaries where parametric equations fail and evaluates how deep gated sequence networks, decision tree ensembles, and hybrid fusion models resolve fast-commerce demand volatility.

### 2. Methodology & Pipeline Workflow
- **Ingestion:** Extracts 6,435 empirical weekly transaction records across 45 distinct retail locations from the historical Walmart supply chain dataset using a strict 80:20 train-test split.
- **Feature Engineering:** Converts raw timestamps into cyclical day-of-week indices and cross-references external operational and macroeconomic parameters (`Store`, `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`).
- **Multi-Track Evaluation:** Simultaneously routes standardized and unscaled feature matrices through a Parametric Baseline (Linear Regression), Non-Parametric Ensembles (Random Forest & Gradient Boosting), a vectorized Deep Sequence Network (LSTM), and a weighted Hybrid Architecture (0.75 * RF + 0.25 * LSTM).

### 3. Empirical Key Findings

| Architectural Domain | Evaluated Model | MAE ($) | RMSE ($) | R2 Score (Accuracy) | Academic Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Parametric Framework | Linear Regression | $433,274.88 | $523,884.74 | 14.8% (0.1481) | **Fail:** Assumes a flat linear trend; cannot map localized dark-store volatile patterns. |
| Deep Sequence Network | Deep LSTM | $164,479.44 | $242,725.71 | 81.7% (0.8171) | **Strong:** Gated memory cells capture non-linear demand shifts, outperforming linear baselines by +66.9%. |
| Non-Parametric Ensemble | Gradient Boosting | $138,905.09 | $200,974.20 | 87.5% (0.8746) | **Optimal:** Progressively minimizes residual variance using sequential boosting steps. |
| Hybrid Architecture | Hybrid (RF + LSTM) | $86,743.67 | $150,458.00 | 93.0% (0.9297) | **Robust:** Fuses neural gating with ensemble partitioning, explaining ~93% of target variance. |
| Non-Parametric Ensemble | Random Forest | $76,582.85 | $147,204.06 | 93.3% (0.9327) | **Superior:** Achieves peak tabular precision via parallelized bootstrap aggregation across 45 stores. |

### 4. Conclusion & Operational Impact
By integrating non-parametric tree ensembles and deep sequence modeling into a decoupled, modular pipeline, supply chain networks can forecast upcoming demand volumes with over 93% precision. This architectural design provides a scalable blueprint for microservices-driven inventory engines—isolating heavy model inference from core transactional services while neutralizing warehouse overstock overhead and stabilizing regional product availability.