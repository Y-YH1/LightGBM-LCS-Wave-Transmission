# -*- coding: utf-8 -*-
"""
Reproducible XGBoost benchmark for prediction of the wave-transmission
coefficient Kt.

"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split


# =============================================================================
# 1. PATHS AND REPRODUCIBILITY SETTINGS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Database.csv"
OUTPUT_DIR = BASE_DIR / "XGBoost_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Target and seven input variables used in the manuscript.
OUTPUT_COL = "Kt"
INPUT_COLS = [
    "RcHm0",   # Rc/Hm0
    "BL",      # Bc/L
    "DAHm0",   # Dn50A/Hm0
    "Sop",     # wave steepness
    "hchd",    # hc/hs
    "Hm0Hd",   # Hm0/hs
    "DCHm0",   # Dn50C/Hm0
]

# Data split used for all benchmark models.
TEST_FRACTION = 0.20
VALIDATION_FRACTION_OF_REMAINDER = 0.25  # 0.25 x 0.80 = 0.20 overall

# Five-fold cross-validation.
N_CV_FOLDS = 5

# Search-stage settings. These are fixed during the staged grid search.
SEARCH_LEARNING_RATE = 0.03
SEARCH_N_ESTIMATORS = 500

# Final-training settings.
FINAL_LEARNING_RATE = 0.02
MAX_FINAL_N_ESTIMATORS = 2500
EARLY_STOPPING_ROUNDS = 50

# Re-run optimization/training by default for reproducibility.
RETRAIN_HYPERPARAMETERS = True
RETRAIN_FINAL_MODEL = True

# CPU settings: parallelize the outer CV search, while keeping each XGBoost
# estimator single-threaded to avoid nested oversubscription.
TOTAL_CPU = multiprocessing.cpu_count()
N_JOBS_SEARCH = max(1, min(8, TOTAL_CPU // 2))
N_JOBS_MODEL = max(1, min(8, TOTAL_CPU // 2))


# =============================================================================
# 2. HYPERPARAMETER SEARCH SPACE
# =============================================================================
# These ranges reproduce the XGBoost benchmark used for the manuscript run.
SEARCH_GRIDS = {
    "tree_complexity": {
        "max_depth": [2, 3, 4],
        "min_child_weight": [3, 5, 8, 12],
    },
    "sampling": {
        "subsample": [0.65, 0.75, 0.85, 0.95],
        "colsample_bytree": [0.65, 0.75, 0.85, 0.95],
    },
    "split_control": {
        "gamma": [0.01, 0.02, 0.05, 0.10, 0.20],
    },
    "regularization": {
        "reg_alpha": [0.01, 0.05, 0.10, 0.20, 0.50],
        "reg_lambda": [2.0, 5.0, 10.0, 20.0],
    },
}

# Fixed parameters during the staged grid search.
INITIAL_PARAMS = {
    "n_estimators": SEARCH_N_ESTIMATORS,
    "learning_rate": SEARCH_LEARNING_RATE,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "gamma": 0.02,
    "reg_alpha": 0.05,
    "reg_lambda": 5.0,
    "max_bin": 128,
    "grow_policy": "depthwise",
}


# =============================================================================
# 3. PLOTTING SETTINGS
# =============================================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16,
    "axes.unicode_minus": False,
})


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================
def read_csv_robust(file_path: Path) -> pd.DataFrame:
    """Read the database using common encodings."""
    encodings = ("utf-8-sig", "utf-8", "gbk")
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise RuntimeError(
        f"Unable to read {file_path}. Tried encodings: {encodings}. "
        f"Last error: {last_error}"
    )


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return R2, MAE, MSE, and RMSE."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
    }


def save_grid_results(grid_search: GridSearchCV, output_path: Path) -> None:
    """Save the complete GridSearchCV table, including mean CV RMSE."""
    result_df = pd.DataFrame(grid_search.cv_results_).copy()
    result_df["mean_CV_RMSE"] = np.sqrt(-result_df["mean_test_score"])
    result_df = result_df.sort_values("rank_test_score")
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def run_grid_step(
    step_name: str,
    base_params: dict,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: KFold,
) -> tuple[dict, float]:
    """Optimize one hyperparameter block using training-only five-fold CV."""
    estimator = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
        **base_params,
    )

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=N_JOBS_SEARCH,
        return_train_score=False,
        pre_dispatch="2*n_jobs",
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_rmse = float(np.sqrt(-search.best_score_))
    save_grid_results(search, OUTPUT_DIR / f"grid_{step_name}.csv")

    print(f"\n{step_name}")
    print("Best parameters:", search.best_params_)
    print(f"Best 5-fold CV RMSE: {best_rmse:.6f}")

    return search.best_params_, best_rmse


def save_split_membership(
    row_ids: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Save exact sample membership for transparent reproduction."""
    subset = np.full(len(row_ids), "", dtype=object)
    subset[train_idx] = "Training"
    subset[val_idx] = "Validation"
    subset[test_idx] = "Testing"

    pd.DataFrame({
        "processed_row_index": np.arange(len(row_ids)),
        "original_csv_row_index": row_ids,
        "subset": subset,
    }).to_csv(
        OUTPUT_DIR / "data_split_membership.csv",
        index=False,
        encoding="utf-8-sig",
    )


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    output_path: Path,
) -> None:
    """Observed-versus-predicted plot for the independent testing subset."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    ax.scatter(
        y_true,
        y_pred,
        s=38,
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        label="Test set",
    )

    all_values = np.concatenate([np.asarray(y_true), np.asarray(y_pred)])
    lower = min(0.0, float(np.nanmin(all_values)))
    upper = max(1.0, float(np.nanmax(all_values)) * 1.05)
    x_ref = np.linspace(lower, upper, 300)

    ax.plot(x_ref, x_ref, "-", linewidth=1.4, label="1:1 line")
    ax.plot(x_ref, 1.2 * x_ref, "--", linewidth=1.2, label="±20% deviation")
    ax.plot(x_ref, 0.8 * x_ref, "--", linewidth=1.2)

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"$K_{\mathrm{t,obs}}\ \mathrm{[-]}$", fontsize=20)
    ax.set_ylabel(r"$K_{\mathrm{t,pred}}\ \mathrm{[-]}$", fontsize=20)
    ax.tick_params(axis="both", direction="in", length=6, width=1.2)

    ax.text(
        0.05,
        0.95,
        f"Test R$^2$ = {metrics['R2']:.4f}\n"
        f"Test RMSE = {metrics['RMSE']:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=13)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(
    evals_result: dict,
    best_iteration_zero_based: int | None,
    output_path: Path,
) -> None:
    """Plot training and validation RMSE versus boosting iteration."""
    train_rmse = evals_result["validation_0"]["rmse"]
    val_rmse = evals_result["validation_1"]["rmse"]
    iterations = np.arange(1, len(train_rmse) + 1)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.plot(iterations, train_rmse, linewidth=1.5, label="Training RMSE")
    ax.plot(iterations, val_rmse, linewidth=1.5, label="Validation RMSE")

    if best_iteration_zero_based is not None:
        effective_iteration = best_iteration_zero_based + 1
        ax.axvline(
            effective_iteration,
            linestyle="--",
            linewidth=1.2,
            label=f"Selected iteration = {effective_iteration}",
        )

    ax.set_xlabel("Boosting iteration", fontsize=18)
    ax.set_ylabel("RMSE", fontsize=18)
    ax.tick_params(axis="both", direction="in", length=6, width=1.2)
    ax.legend(frameon=False, fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 5. MAIN PROGRAM
# =============================================================================
def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DATA_PATH}\n"
            "Place 'Database.csv' in the same directory as this script."
        )

    # -------------------------------------------------------------------------
    # 5.1 Read and validate data
    # -------------------------------------------------------------------------
    data_raw = read_csv_robust(DATA_PATH)
    required_cols = INPUT_COLS + [OUTPUT_COL]

    missing_columns = [c for c in required_cols if c not in data_raw.columns]
    if missing_columns:
        raise KeyError("Missing required columns: " + ", ".join(missing_columns))

    # Preserve original CSV row indices before removing incomplete records.
    data = data_raw[required_cols].copy()
    data["_original_row_index"] = data.index.to_numpy()
    n_original = len(data)
    data = data.dropna(subset=required_cols).reset_index(drop=True)

    print("Original sample size:", n_original)
    print("Valid sample size:", len(data))
    print("Removed incomplete records:", n_original - len(data))

    X_all = data[INPUT_COLS].to_numpy(dtype=float)
    y_all = data[OUTPUT_COL].to_numpy(dtype=float).ravel()
    row_ids = data["_original_row_index"].to_numpy(dtype=int)

    # -------------------------------------------------------------------------
    # 5.2 Reproducible 60/20/20 split
    # -------------------------------------------------------------------------
    all_idx = np.arange(len(data))
    temp_idx, test_idx = train_test_split(
        all_idx,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    train_idx, val_idx = train_test_split(
        temp_idx,
        test_size=VALIDATION_FRACTION_OF_REMAINDER,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    save_split_membership(row_ids, train_idx, val_idx, test_idx)

    split_summary = pd.DataFrame({
        "Subset": ["Training", "Validation", "Testing", "Total"],
        "N": [len(train_idx), len(val_idx), len(test_idx), len(all_idx)],
    })
    split_summary.to_csv(
        OUTPUT_DIR / "data_split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n", split_summary.to_string(index=False), sep="")

    # -------------------------------------------------------------------------
    # 5.3 Training-only five-fold CV
    # -------------------------------------------------------------------------
    cv = KFold(
        n_splits=N_CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    best_params_file = OUTPUT_DIR / "xgboost_selected_params.json"

    if RETRAIN_HYPERPARAMETERS or not best_params_file.exists():
        tuned_params = INITIAL_PARAMS.copy()
        cv_summary = []

        for step_name, grid in SEARCH_GRIDS.items():
            best_step, best_rmse = run_grid_step(
                step_name,
                tuned_params,
                grid,
                X_train,
                y_train,
                cv,
            )
            tuned_params.update(best_step)
            cv_summary.append({
                "Step": step_name,
                "Best_CV_RMSE": best_rmse,
                "Best_parameters": json.dumps(best_step, sort_keys=True),
            })

        # The learning rate is reduced for final training; this value is a
        # predefined final-training setting rather than a grid-searched value.
        tuned_params["learning_rate"] = FINAL_LEARNING_RATE

        # The final number of boosting rounds is selected by validation-based
        # early stopping, not by the fixed search-stage n_estimators value.
        tuned_params.pop("n_estimators", None)

        with open(best_params_file, "w", encoding="utf-8") as file:
            json.dump(tuned_params, file, indent=4, ensure_ascii=False)

        pd.DataFrame(cv_summary).to_csv(
            OUTPUT_DIR / "xgboost_CV_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        with open(best_params_file, "r", encoding="utf-8") as file:
            tuned_params = json.load(file)

    print("\nSelected XGBoost parameters for final training:")
    for key, value in tuned_params.items():
        print(f"  {key}: {value}")

    # Save a transparent description of all search ranges/strategies.
    search_description = {
        "search_stage_learning_rate": SEARCH_LEARNING_RATE,
        "search_stage_n_estimators": SEARCH_N_ESTIMATORS,
        "search_grids": SEARCH_GRIDS,
        "final_learning_rate": FINAL_LEARNING_RATE,
        "max_final_n_estimators": MAX_FINAL_N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    }
    with open(
        OUTPUT_DIR / "xgboost_search_strategy.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(search_description, file, indent=4, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # 5.4 Final XGBoost model
    # -------------------------------------------------------------------------
    model_json_path = OUTPUT_DIR / "xgboost_Kt_model.json"
    model_pkl_path = OUTPUT_DIR / "xgboost_Kt_model.pkl"

    if RETRAIN_FINAL_MODEL or not model_json_path.exists():
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            eval_metric="rmse",
            n_estimators=MAX_FINAL_N_ESTIMATORS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS_MODEL,
            verbosity=0,
            **tuned_params,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        model.save_model(model_json_path)
        joblib.dump(model, model_pkl_path)
    else:
        model = xgb.XGBRegressor()
        model.load_model(model_json_path)

    best_iteration_zero_based = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    effective_n_estimators = (
        None
        if best_iteration_zero_based is None
        else int(best_iteration_zero_based + 1)
    )

    print("\nFinal XGBoost model")
    print("Effective boosting iterations:", effective_n_estimators)
    if best_score is not None:
        print(f"Best validation RMSE: {float(best_score):.6f}")

    final_report = {
        "selected_hyperparameters": tuned_params,
        "search_stage_learning_rate": SEARCH_LEARNING_RATE,
        "search_stage_n_estimators": SEARCH_N_ESTIMATORS,
        "final_learning_rate": FINAL_LEARNING_RATE,
        "maximum_final_n_estimators": MAX_FINAL_N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "effective_n_estimators": effective_n_estimators,
        "best_validation_RMSE": None if best_score is None else float(best_score),
        "random_state": RANDOM_STATE,
    }
    with open(
        OUTPUT_DIR / "xgboost_final_parameter_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_report, file, indent=4, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # 5.5 Independent performance evaluation
    # -------------------------------------------------------------------------
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    metrics_train = evaluate_regression(y_train, pred_train)
    metrics_val = evaluate_regression(y_val, pred_val)
    metrics_test = evaluate_regression(y_test, pred_test)

    metrics_df = pd.DataFrame([
        {"Model": "XGBoost", "Subset": "Training", **metrics_train},
        {"Model": "XGBoost", "Subset": "Validation", **metrics_val},
        {"Model": "XGBoost", "Subset": "Testing", **metrics_test},
    ])
    metrics_df.to_csv(
        OUTPUT_DIR / "xgboost_performance_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\nPerformance metrics")
    print(metrics_df.to_string(index=False))

    prediction_df = pd.DataFrame(X_test, columns=INPUT_COLS)
    prediction_df["Kt_observed"] = y_test
    prediction_df["Kt_predicted_XGBoost"] = pred_test
    prediction_df["Residual_XGBoost"] = y_test - pred_test
    prediction_df["processed_row_index"] = test_idx
    prediction_df["original_csv_row_index"] = row_ids[test_idx]
    prediction_df.to_csv(
        OUTPUT_DIR / "xgboost_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 5.6 Figures and learning-curve data
    # -------------------------------------------------------------------------
    plot_prediction_scatter(
        y_test,
        pred_test,
        metrics_test,
        OUTPUT_DIR / "XGBoost_test_scatter.png",
    )

    evals_result = model.evals_result()
    plot_training_curve(
        evals_result,
        best_iteration_zero_based,
        OUTPUT_DIR / "XGBoost_training_validation_RMSE.png",
    )

    pd.DataFrame({
        "Iteration": np.arange(
            1,
            len(evals_result["validation_0"]["rmse"]) + 1,
        ),
        "Training_RMSE": evals_result["validation_0"]["rmse"],
        "Validation_RMSE": evals_result["validation_1"]["rmse"],
    }).to_csv(
        OUTPUT_DIR / "XGBoost_training_validation_RMSE.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
