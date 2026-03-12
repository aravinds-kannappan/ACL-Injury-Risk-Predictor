"""Model evaluation with comprehensive metrics and visualizations."""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Compute all evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    logger.info(
        f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
        f"ROC AUC: {metrics['roc_auc']:.4f}, "
        f"F1: {metrics['f1']:.4f}"
    )
    return metrics


def plot_roc_curve(
    y_test: np.ndarray, y_proba: np.ndarray,
    model_name: str, save_path: Path = None
) -> plt.Figure:
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"{model_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve - {model_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_test: np.ndarray, y_pred: np.ndarray,
    model_name: str, save_path: Path = None
) -> plt.Figure:
    """Plot confusion matrix as heatmap."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Healthy", "ACL Injured"],
        yticklabels=["Healthy", "ACL Injured"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    model, feature_names: list, top_n: int = 15, save_path: Path = None
) -> plt.Figure:
    """Plot top feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model does not have feature importances.")
        return None

    # Get top N
    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(range(len(top_features)), top_importances, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_evaluation_report(
    training_results: dict, output_dir: Path = None
) -> dict:
    """Generate full evaluation report with plots for all models."""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    X_test = training_results["X_test"]
    y_test = training_results["y_test"]
    report = {}

    for model_name in ["random_forest", "logistic_regression"]:
        if model_name not in training_results:
            continue

        model = training_results[model_name]["model"]
        metrics = evaluate_model(model, X_test, y_test, model_name)
        report[model_name] = metrics

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        plot_roc_curve(
            y_test, y_proba, model_name,
            output_dir / f"{model_name}_roc.png",
        )
        plot_confusion_matrix(
            y_test, y_pred, model_name,
            output_dir / f"{model_name}_confusion.png",
        )

        from src.features.feature_pipeline import FEATURE_NAMES
        plot_feature_importance(
            model, FEATURE_NAMES, top_n=15,
            save_path=output_dir / f"{model_name}_feature_importance.png",
        )

    return report
