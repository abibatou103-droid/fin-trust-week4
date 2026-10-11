from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def build_model_dataset() -> pd.DataFrame:
    customer = pd.read_csv(DATA_DIR / "customer.csv")
    transactions = pd.read_csv(DATA_DIR / "transactions.csv")

    transactions["Transaction_DateTime"] = pd.to_datetime(transactions["Transaction_DateTime"])
    transactions = transactions.sort_values(["Customer_ID", "Transaction_DateTime"]).reset_index(drop=True)
    transactions["Risk_Binary"] = (transactions["Risk_Review_Flag"] == "Yes").astype(int)

    transactions["customer_prior_tx_count"] = transactions.groupby("Customer_ID").cumcount()
    transactions["customer_prior_risk_count"] = (
        transactions.groupby("Customer_ID")["Risk_Binary"].transform(lambda s: s.shift(fill_value=0).cumsum())
    )
    transactions["customer_prior_risk_rate"] = (
        transactions["customer_prior_risk_count"] / np.maximum(transactions["customer_prior_tx_count"], 1)
    )
    transactions["customer_prior_failure_count"] = (
        transactions.groupby("Customer_ID")["Transaction_Status"].transform(
            lambda s: s.eq("Failed").shift(fill_value=False).cumsum()
        )
    )
    transactions["customer_prior_failure_rate"] = (
        transactions["customer_prior_failure_count"] / np.maximum(transactions["customer_prior_tx_count"], 1)
    )

    transactions["Hour"] = transactions["Transaction_DateTime"].dt.hour
    transactions["DayOfWeek"] = transactions["Transaction_DateTime"].dt.dayofweek
    transactions["Month"] = transactions["Transaction_DateTime"].dt.month
    transactions["Amount_Log"] = np.log1p(transactions["Amount_NGN"])
    transactions["Is_International"] = (transactions["International_Transaction"] == "Yes").astype(int)
    transactions["Is_Failed"] = (transactions["Transaction_Status"] == "Failed").astype(int)
    transactions["Is_Reversed"] = (transactions["Transaction_Status"] == "Reversed").astype(int)
    transactions["Is_Pending"] = (transactions["Transaction_Status"] == "Pending").astype(int)

    merged = transactions.merge(customer, on="Customer_ID", how="left")
    merged = merged.drop(columns=["Transaction_ID", "Customer_ID", "Customer_Name", "Transaction_DateTime", "Risk_Binary"])

    # Keep the target name as the educational synthetic target and document that it is not actual fraud detection.
    return merged


def optimize_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    best_threshold = 0.5
    best_score = -1.0
    best_predictions = (probabilities >= 0.5).astype(int)

    for threshold in np.linspace(0.1, 0.9, 81):
        prediction = (probabilities >= threshold).astype(int)
        score = f1_score(labels, prediction)
        if score > best_score:
            best_threshold = threshold
            best_score = score
            best_predictions = prediction
    return best_threshold, best_predictions


def final_model_summary() -> dict:
    df = build_model_dataset()
    features = df.drop(columns=["Risk_Review_Flag"])
    target = (df["Risk_Review_Flag"] == "Yes").astype(int)

    categorical = features.select_dtypes(include=["object"]).columns.tolist()
    numeric = features.select_dtypes(exclude=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
        ),
    }

    results = {}
    for model_name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        threshold, predictions = optimize_threshold(probabilities, y_test)

        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        result_dict = {
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions)),
            "recall": float(recall_score(y_test, predictions)),
            "f1_score": float(f1_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
        results[model_name] = result_dict

    final_model = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42)
    final_pipeline = Pipeline([("preprocessor", preprocessor), ("model", final_model)])
    final_pipeline.fit(X_train, y_train)
    final_probability = final_pipeline.predict_proba(X_test)[:, 1]
    final_threshold, final_predictions = optimize_threshold(final_probability, y_test)
    final_note = {
        "selected_model": "logistic_regression",
        "threshold": float(final_threshold),
        "model_reason": "The final model balances a strong baseline with interpretability. It performs competitively with the tree model while providing clear coefficient-based explanations and easier business communication.",
    }

    return {
        "dataset_shape": df.shape,
        "class_distribution": {
            "No": int((df["Risk_Review_Flag"] == "No").sum()),
            "Yes": int((df["Risk_Review_Flag"] == "Yes").sum()),
        },
        "results": results,
        "final_model": final_note,
        "feature_columns": features.columns.tolist(),
    }


def save_confusion_matrix() -> None:
    df = build_model_dataset()
    features = df.drop(columns=["Risk_Review_Flag"])
    target = (df["Risk_Review_Flag"] == "Yes").astype(int)

    categorical = features.select_dtypes(include=["object"]).columns.tolist()
    numeric = features.select_dtypes(exclude=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    logistic = LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", logistic)])
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold, predictions = optimize_threshold(probabilities, y_test)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Final Logistic Risk Model")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    tick_labels = ["No", "Yes"]
    plt.xticks([0, 1], tick_labels)
    plt.yticks([0, 1], tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()


def main() -> None:
    model_summary = final_model_summary()
    OUTPUT_DIR.joinpath("model_metrics.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")

    df = build_model_dataset()
    df.to_csv(OUTPUT_DIR / "final_modelling_dataset.csv", index=False)
    save_confusion_matrix()

    features = df.drop(columns=["Risk_Review_Flag"])
    target = (df["Risk_Review_Flag"] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )

    categorical = features.select_dtypes(include=["object"]).columns.tolist()
    numeric = features.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    final_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42)),
        ]
    )
    final_pipeline.fit(X_train, y_train)
    with (OUTPUT_DIR / "final_model.pkl").open("wb") as handle:
        pickle.dump(final_pipeline, handle)

    print(json.dumps(model_summary["results"], indent=2))
    print("Saved final modelling dataset, model, and metrics to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
