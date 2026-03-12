"""Static visualizations for biomechanical analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_joint_angles_over_gait_cycle(
    angles_dict: dict, title: str = "Joint Angles Over Gait Cycle",
    save_path: Path = None
) -> plt.Figure:
    """Plot joint angle time series vs gait cycle percentage.

    Args:
        angles_dict: Dict mapping label strings to arrays of shape (101,)
            or (n_subjects, 101) for mean +/- SD.
        title: Plot title.
        save_path: Optional path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    gait_pct = np.linspace(0, 100, 101)
    colors = plt.cm.tab10(np.linspace(0, 1, len(angles_dict)))

    for (label, data), color in zip(angles_dict.items(), colors):
        data = np.asarray(data)
        if data.ndim == 2:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            ax.plot(gait_pct, mean, color=color, lw=2, label=label)
            ax.fill_between(gait_pct, mean - std, mean + std, color=color, alpha=0.2)
        else:
            ax.plot(gait_pct, data, color=color, lw=2, label=label)

    ax.set_xlabel("Gait Cycle (%)", fontsize=12)
    ax.set_ylabel("Angle (degrees)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_distributions(
    X: np.ndarray, y: np.ndarray, feature_names: list,
    top_n: int = 8, save_path: Path = None
) -> plt.Figure:
    """Violin plots comparing healthy vs injured distributions."""
    import pandas as pd

    # Find features with most separation
    healthy = X[y == 0]
    injured = X[y == 1]

    if len(healthy) == 0 or len(injured) == 0:
        return None

    # Use t-statistic as proxy for separation
    separations = []
    for i in range(X.shape[1]):
        h_std = np.std(healthy[:, i]) + 1e-8
        i_std = np.std(injured[:, i]) + 1e-8
        pooled_std = np.sqrt((h_std**2 + i_std**2) / 2)
        t = abs(np.mean(healthy[:, i]) - np.mean(injured[:, i])) / pooled_std
        separations.append(t)

    top_indices = np.argsort(separations)[-top_n:][::-1]

    fig, axes = plt.subplots(2, top_n // 2, figsize=(16, 8))
    axes = axes.flatten()

    for ax_idx, feat_idx in enumerate(top_indices):
        if ax_idx >= len(axes):
            break

        data = pd.DataFrame({
            "Value": np.concatenate([healthy[:, feat_idx], injured[:, feat_idx]]),
            "Group": ["Healthy"] * len(healthy) + ["ACL Injured"] * len(injured),
        })

        sns.violinplot(data=data, x="Group", y="Value", ax=axes[ax_idx], palette=["#4CAF50", "#F44336"])
        axes[ax_idx].set_title(feature_names[feat_idx], fontsize=9)
        axes[ax_idx].set_xlabel("")

    fig.suptitle("Feature Distributions: Healthy vs ACL Injured", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_matrix(
    X: np.ndarray, feature_names: list, save_path: Path = None
) -> plt.Figure:
    """Heatmap of feature correlations."""
    import pandas as pd

    # Use top 20 features by variance for readability
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[-20:]

    df = pd.DataFrame(X[:, top_indices], columns=[feature_names[i] for i in top_indices])
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, fmt=".1f",
                xticklabels=True, yticklabels=True)
    ax.set_title("Feature Correlation Matrix (Top 20 by Variance)", fontsize=14)
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_angle_comparison(
    healthy_angles: np.ndarray, injured_angles: np.ndarray,
    joint_name: str, save_path: Path = None
) -> plt.Figure:
    """Overlay gait cycle plots for healthy vs injured groups."""
    fig, ax = plt.subplots(figsize=(10, 6))
    gait_pct = np.linspace(0, 100, healthy_angles.shape[-1])

    for label, data, color in [
        ("Healthy", healthy_angles, "#4CAF50"),
        ("ACL Injured", injured_angles, "#F44336"),
    ]:
        if data.ndim == 2:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            ax.plot(gait_pct, mean, color=color, lw=2, label=f"{label} (n={len(data)})")
            ax.fill_between(gait_pct, mean - std, mean + std, color=color, alpha=0.15)
        else:
            ax.plot(gait_pct, data, color=color, lw=2, label=label)

    ax.set_xlabel("Gait Cycle (%)", fontsize=12)
    ax.set_ylabel(f"{joint_name} Angle (degrees)", fontsize=12)
    ax.set_title(f"{joint_name}: Healthy vs ACL Injured", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
