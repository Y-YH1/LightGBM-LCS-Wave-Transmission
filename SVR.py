# -*- coding: utf-8 -*-
"""
SVR benchmark model for wave transmission coefficient Kt


"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    make_scorer,
)

# ============================================================
# 1. Reproducible settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Database.csv"
OUTPUT_DIR = BASE_DIR / "SVR_results"

OUTPUT_COL = "Kt"

INPUT_COLS = [
    "RcHm0",
    "BL",
    "DAHm0",
    "Sop",
    "hchd",
    "Hm0Hd",
    "DCHm0",
]

RANDOM_STATE = 42
N_SPLITS = 5
N_BOOTSTRAP = 2000


# ============================================================
# 2. Utility functions
# ============================================================

def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read the database using common encodings."""
    encodings = ("utf-8-sig", "utf-8", "gbk")
    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding)
            print(f"Database loaded with encoding: {encoding}")
            return df
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to read {path.name}. Tried encodings: {encodings}. "
        f"Last error: {last_error}"
    )


def rmse_value(y_true, y_pred) -> float:
    """Version-independent RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def negative_rmse_scorer(y_true, y_pred) -> float:
    """RMSE scorer used with greater_is_better=False."""
    return rmse_value(y_true, y_pred)


RMSE_SCORER = make_scorer(
    negative_rmse_scorer,
    greater_is_better=False,
)


def evaluate_regression(y_true, y_pred, subset_name: str) -> dict:
    """Calculate R2, MAE, MSE, and RMSE."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mse = float(mean_squared_error(y_true, y_pred))
    metrics = {
        "Subset": subset_name,
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }

    print(
        f"{subset_name:10s} | "
        f"R2 = {metrics['R2']:.6f} | "
        f"RMSE = {metrics['RMSE']:.6f} | "
        f"MAE = {metrics['MAE']:.6f}"
    )
    return metrics


def paired_bootstrap_ci(
    y_true,
    y_pred,
    n_bootstrap: int = 2000,
    random_state: int = 42,
    confidence: float = 0.95,
) -> dict:
    """
    Nonparametric paired bootstrap on the independent test subset.

    The observed-predicted pairs are resampled together so that the
    correspondence between each observation and prediction is preserved.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    rng = np.random.default_rng(random_state)
    n = len(y_true)

    r2_values = np.empty(n_bootstrap, dtype=float)
    rmse_values = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]

        r2_values[b] = r2_score(yt, yp)
        rmse_values[b] = rmse_value(yt, yp)

    alpha = 1.0 - confidence
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    return {
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": float(confidence),
        "R2_lower": float(np.percentile(r2_values, lower_q)),
        "R2_upper": float(np.percentile(r2_values, upper_q)),
        "RMSE_lower": float(np.percentile(rmse_values, lower_q)),
        "RMSE_upper": float(np.percentile(rmse_values, upper_q)),
    }


def save_test_scatter(y_true, y_pred, metrics: dict, save_path: Path) -> None:
    """Observed-versus-predicted test-set scatter plot."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=300)

    ax.scatter(
        y_true,
        y_pred,
        s=36,
        facecolors="none",
        edgecolors="black",
        linewidths=0.9,
        label="Test set",
    )

    ref = np.linspace(0.0, 1.0, 300)
    ax.plot(ref, ref, "-", linewidth=1.3, label="1:1 line")
    ax.plot(ref, 1.2 * ref, "--", linewidth=1.1, label="±20% deviation")
    ax.plot(ref, 0.8 * ref, "--", linewidth=1.1)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$K_{\mathrm{t,obs}}\ \mathrm{[-]}$")
    ax.set_ylabel(r"$K_{\mathrm{t,pred}}\ \mathrm{[-]}$")
    ax.tick_params(axis="both", direction="in")

    ax.text(
        0.05,
        0.95,
        f"SVR Test R$^2$ = {metrics['R2']:.4f}\n"
        f"SVR Test RMSE = {metrics['RMSE']:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()

    fig.savefig(save_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3. Main workflow
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Database file not found: {DATA_PATH}\n"
            "Please place Database.csv in the same directory as this script."
        )

    # --------------------------------------------------------
    # 3.1 Read and validate data
    # --------------------------------------------------------
    data_raw = read_csv_robust(DATA_PATH)

    required_cols = INPUT_COLS + [OUTPUT_COL]
    missing_cols = [c for c in required_cols if c not in data_raw.columns]

    if missing_cols:
        raise KeyError(
            "The following required columns are missing from Database.csv:\n"
            + ", ".join(missing_cols)
        )

    # Retain only required model columns and remove missing records.
    data = data_raw[required_cols].dropna().reset_index(drop=True)

    X_all = data[INPUT_COLS].to_numpy(dtype=float)
    y_all = data[OUTPUT_COL].to_numpy(dtype=float).ravel()

    if not np.isfinite(X_all).all():
        raise ValueError("Input matrix contains inf or -inf values.")
    if not np.isfinite(y_all).all():
        raise ValueError("Target vector contains inf or -inf values.")

    print(f"Original sample size: {len(data_raw)}")
    print(f"Valid sample size:    {len(data)}")
    print(f"Removed samples:      {len(data_raw) - len(data)}")

    # --------------------------------------------------------
    # 3.2 Reproduce the manuscript split: 60 / 20 / 20
    # --------------------------------------------------------
    all_indices = np.arange(len(data))

    idx_temp, idx_test = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    idx_train, idx_val = train_test_split(
        idx_temp,
        test_size=0.25,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_val, y_val = X_all[idx_val], y_all[idx_val]
    X_test, y_test = X_all[idx_test], y_all[idx_test]

    print("\nDataset split")
    print(f"Training:   {len(idx_train)}")
    print(f"Validation: {len(idx_val)}")
    print(f"Testing:    {len(idx_test)}")
    print(f"Total:      {len(data)}")

    split_df = pd.concat(
        [
            pd.DataFrame({"Original_row_index": idx_train, "Subset": "Training"}),
            pd.DataFrame({"Original_row_index": idx_val, "Subset": "Validation"}),
            pd.DataFrame({"Original_row_index": idx_test, "Subset": "Testing"}),
        ],
        ignore_index=True,
    )
    split_df.to_csv(
        OUTPUT_DIR / "SVR_data_split_indices.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 3.3 Five-fold CV on the TRAINING subset only
    # --------------------------------------------------------
    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # StandardScaler is deliberately inside the Pipeline so that
    # scaling is fitted independently within each CV training fold.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR()),
        ]
    )

    # Search ranges used in the manuscript.
    param_grid = [
        {
            "svr__kernel": ["rbf"],
            "svr__C": [1, 10, 100, 1000],
            "svr__gamma": ["scale", 0.01, 0.03, 0.1, 0.3, 1.0],
            "svr__epsilon": [0.005, 0.01, 0.02, 0.05],
        },
        {
            "svr__kernel": ["linear"],
            "svr__C": [0.1, 1, 10, 100],
            "svr__epsilon": [0.005, 0.01, 0.02, 0.05],
        },
    ]

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=RMSE_SCORER,
        cv=cv,
        refit=True,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    # Validation and test subsets are not used here.
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_cv_rmse = -float(search.best_score_)

    clean_best_params = {
        key.replace("svr__", ""): value
        for key, value in search.best_params_.items()
    }

    print("\nBest SVR hyperparameters")
    for key, value in clean_best_params.items():
        print(f"{key}: {value}")
    print(f"Best 5-fold CV RMSE: {best_cv_rmse:.6f}")

    with open(
        OUTPUT_DIR / "SVR_best_parameters.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "best_parameters": clean_best_params,
                "best_5fold_cv_rmse": best_cv_rmse,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["mean_test_RMSE"] = -cv_results["mean_test_score"]
    cv_results["mean_train_RMSE"] = -cv_results["mean_train_score"]
    cv_results = cv_results.sort_values("rank_test_score")

    cv_results.to_csv(
        OUTPUT_DIR / "SVR_grid_search_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 3.4 Save optimized SVR model
    # --------------------------------------------------------
    joblib.dump(
        best_model,
        OUTPUT_DIR / "SVR_model_optimized.pkl",
    )

    # --------------------------------------------------------
    # 3.5 Evaluate train / validation / test subsets
    # --------------------------------------------------------
    pred_train = best_model.predict(X_train)
    pred_val = best_model.predict(X_val)
    pred_test = best_model.predict(X_test)

    print("\nModel performance")
    metrics_train = evaluate_regression(y_train, pred_train, "Training")
    metrics_val = evaluate_regression(y_val, pred_val, "Validation")
    metrics_test = evaluate_regression(y_test, pred_test, "Testing")

    metrics_df = pd.DataFrame(
        [metrics_train, metrics_val, metrics_test]
    )
    metrics_df.to_csv(
        OUTPUT_DIR / "SVR_performance_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 3.6 Test-set paired bootstrap confidence intervals
    # --------------------------------------------------------
    bootstrap_ci = paired_bootstrap_ci(
        y_true=y_test,
        y_pred=pred_test,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE,
        confidence=0.95,
    )

    with open(
        OUTPUT_DIR / "SVR_test_bootstrap_95CI.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(bootstrap_ci, f, indent=4)

    print(
        "\nTest-set 95% bootstrap confidence intervals "
        f"({N_BOOTSTRAP} paired resamples)"
    )
    print(
        f"R2:   ({bootstrap_ci['R2_lower']:.4f}, "
        f"{bootstrap_ci['R2_upper']:.4f})"
    )
    print(
        f"RMSE: ({bootstrap_ci['RMSE_lower']:.4f}, "
        f"{bootstrap_ci['RMSE_upper']:.4f})"
    )

    # --------------------------------------------------------
    # 3.7 Save independent test predictions
    # --------------------------------------------------------
    test_predictions = data.iloc[idx_test][INPUT_COLS].copy()
    test_predictions.insert(0, "Original_row_index", idx_test)
    test_predictions["Kt_observed"] = y_test
    test_predictions["Kt_predicted_SVR"] = pred_test
    test_predictions["Residual_observed_minus_predicted"] = y_test - pred_test
    test_predictions["Absolute_error"] = np.abs(y_test - pred_test)

    test_predictions.to_csv(
        OUTPUT_DIR / "SVR_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 3.8 Reviewer-ready test-set scatter plot
    # --------------------------------------------------------
    save_test_scatter(
        y_true=y_test,
        y_pred=pred_test,
        metrics=metrics_test,
        save_path=OUTPUT_DIR / "SVR_test_scatter",
    )

    # --------------------------------------------------------
    # 3.9 Save concise reproducibility summary
    # --------------------------------------------------------
    summary = {
        "random_state": RANDOM_STATE,
        "split": {
            "training_fraction": 0.60,
            "validation_fraction": 0.20,
            "testing_fraction": 0.20,
        },
        "five_fold_cv_on_training_subset_only": True,
        "standard_scaler_inside_cv_pipeline": True,
        "validation_used_for_hyperparameter_tuning": False,
        "testing_used_for_hyperparameter_tuning": False,
        "best_parameters": clean_best_params,
        "best_5fold_cv_rmse": best_cv_rmse,
        "test_R2": metrics_test["R2"],
        "test_RMSE": metrics_test["RMSE"],
        "bootstrap_95CI": bootstrap_ci,
    }

    with open(
        OUTPUT_DIR / "SVR_reproducibility_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"\nAll outputs were saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
