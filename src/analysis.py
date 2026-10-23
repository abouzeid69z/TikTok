"""Reproduce the TikTok claim-classification analysis and report artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tiktok-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tiktok_dataset.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

NUMERIC_FEATURES = [
    "video_duration_sec",
    "video_view_count",
    "video_like_count",
    "video_share_count",
    "video_download_count",
    "video_comment_count",
    "text_length",
]
CATEGORICAL_FEATURES = ["verified_status", "author_ban_status"]
TARGET = "claim_status"


def load_data() -> pd.DataFrame:
    """Load and validate the columns used by the analysis."""
    data = pd.read_csv(DATA_PATH)
    required = set(NUMERIC_FEATURES[:-1] + CATEGORICAL_FEATURES + [TARGET, "video_transcription_text"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data = data.dropna(subset=sorted(required)).copy()
    data["text_length"] = data["video_transcription_text"].str.len()
    data[TARGET] = data[TARGET].map({"opinion": 0, "claim": 1})
    if data[TARGET].isna().any():
        raise ValueError("Unexpected claim_status label found")
    return data


def save_exploratory_figures(data: pd.DataFrame) -> None:
    """Create the concise visual summary used in the portfolio README."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    labels = data[TARGET].map({0: "Opinion", 1: "Claim"})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=labels, hue=labels, legend=False, palette="Set2", ax=ax)
    ax.set(title="Videos are nearly balanced by label", xlabel="Video label", ylabel="Videos")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)

    plot_data = data.assign(claim_status_label=labels)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(
        data=plot_data,
        x="claim_status_label",
        y="video_view_count",
        hue="claim_status_label",
        legend=False,
        showfliers=False,
        palette="Set2",
        ax=ax,
    )
    ax.set(
        title="Claims receive substantially more views",
        xlabel="Video label",
        ylabel="Video views (outliers hidden for readability)",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "views_by_status.png", dpi=160)
    plt.close(fig)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def metric_summary(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, object]:
    """Tune models only against training folds, with claim recall as the objective."""
    models: dict[str, object] = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                ("model", LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": GridSearchCV(
            Pipeline(
                [
                    ("preprocess", build_preprocessor()),
                    (
                        "model",
                        RandomForestClassifier(
                            class_weight="balanced",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            param_grid={
                "model__n_estimators": [150, 300],
                "model__max_depth": [8, None],
                "model__min_samples_leaf": [1, 3],
            },
            scoring="recall",
            cv=5,
            n_jobs=-1,
        ),
        "xgboost": GridSearchCV(
            Pipeline(
                [
                    ("preprocess", build_preprocessor()),
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            param_grid={
                "model__n_estimators": [150, 300],
                "model__max_depth": [4, 8],
                "model__learning_rate": [0.05, 0.1],
            },
            scoring="recall",
            cv=5,
            n_jobs=-1,
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def save_model_figures(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Opinion", "Claim"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Champion model: held-out test set")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    if not isinstance(estimator, Pipeline) or not hasattr(estimator["model"], "feature_importances_"):
        return
    names = estimator["preprocess"].get_feature_names_out()
    names = [name.split("__", 1)[-1] for name in names]
    importances = pd.Series(estimator["model"].feature_importances_, index=names).nlargest(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    importances.sort_values().plot.barh(ax=ax, color="#3b82f6")
    ax.set(title="Top model features", xlabel="Model feature importance", ylabel="")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)


def main() -> None:
    data = load_data()
    save_exploratory_figures(data)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = data[features]
    y = data[TARGET].astype(int)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.5,
        stratify=y_holdout,
        random_state=RANDOM_STATE,
    )

    models = train_models(X_train, y_train)
    validation_metrics = {
        name: metric_summary(y_validation, model.predict(X_validation))
        for name, model in models.items()
    }
    # F1 avoids selecting a trivial all-positive model that has perfect recall
    # but no practical value. Recall is still reported for moderation triage.
    champion_name = max(validation_metrics, key=lambda name: validation_metrics[name]["f1"])
    champion = models[champion_name]
    test_predictions = champion.predict(X_test)

    verified = data.loc[data["verified_status"] == "verified", "video_view_count"]
    not_verified = data.loc[data["verified_status"] == "not verified", "video_view_count"]
    t_statistic, p_value = ttest_ind(verified, not_verified, equal_var=False)

    report = {
        "data": {
            "raw_rows": int(pd.read_csv(DATA_PATH, usecols=[TARGET]).shape[0]),
            "modeling_rows": int(len(data)),
            "claim_share": float(y.mean()),
        },
        "verification_t_test": {
            "verified_mean_views": float(verified.mean()),
            "not_verified_mean_views": float(not_verified.mean()),
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
        },
        "validation_metrics": validation_metrics,
        "champion_model": champion_name,
        "test_metrics": metric_summary(y_test, test_predictions),
        "test_classification_report": classification_report(
            y_test,
            test_predictions,
            target_names=["opinion", "claim"],
            output_dict=True,
            zero_division=0,
        ),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    save_model_figures(champion, X_test, y_test)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
