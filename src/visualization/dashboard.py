"""Risk score dashboard visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def create_gauge_chart(score: float, ax: plt.Axes = None) -> plt.Axes:
    """Draw a semicircular risk score gauge.

    Green (0-0.3) -> Yellow (0.3-0.7) -> Red (0.7-1.0)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw gauge background arcs
    for start, end, color in [
        (0, 0.3, "#4CAF50"),
        (0.3, 0.7, "#FF9800"),
        (0.7, 1.0, "#F44336"),
    ]:
        theta_start = 180 - start * 180
        theta_end = 180 - end * 180
        arc = patches.Arc(
            (0, 0), 2.2, 2.2,
            angle=0, theta1=theta_end, theta2=theta_start,
            color=color, lw=20, alpha=0.7,
        )
        ax.add_patch(arc)

    # Draw needle
    angle = np.radians(180 - score * 180)
    needle_len = 0.9
    ax.plot(
        [0, needle_len * np.cos(angle)],
        [0, needle_len * np.sin(angle)],
        color="#333333", lw=3, solid_capstyle="round",
    )
    ax.plot(0, 0, "o", color="#333333", markersize=10)

    # Score text
    ax.text(0, -0.15, f"{score:.1%}", fontsize=24, ha="center",
            va="top", fontweight="bold")

    # Labels
    ax.text(-1.1, -0.05, "Low", fontsize=10, ha="center", color="#4CAF50")
    ax.text(0, 1.15, "Moderate", fontsize=10, ha="center", color="#FF9800")
    ax.text(1.1, -0.05, "High", fontsize=10, ha="center", color="#F44336")

    return ax


def create_risk_dashboard(
    risk_score: float,
    risk_level: str,
    contributing_factors: list,
    joint_angles: dict = None,
    save_path: Path = None,
) -> plt.Figure:
    """Create complete risk assessment dashboard.

    Args:
        risk_score: 0.0 to 1.0 risk probability.
        risk_level: "Low", "Moderate", or "High".
        contributing_factors: List of dicts with 'feature', 'importance', 'value'.
        joint_angles: Optional dict of joint angle summaries.
        save_path: Optional path to save figure.
    """
    fig = plt.figure(figsize=(16, 10))

    # Title
    fig.suptitle("ACL Injury Risk Assessment", fontsize=18, fontweight="bold", y=0.98)

    # 1. Gauge chart (top left)
    ax_gauge = fig.add_axes([0.05, 0.45, 0.4, 0.45])
    create_gauge_chart(risk_score, ax_gauge)
    ax_gauge.set_title("Risk Score", fontsize=14, pad=10)

    # 2. Risk level summary (top right)
    ax_summary = fig.add_axes([0.5, 0.45, 0.45, 0.45])
    ax_summary.axis("off")

    level_colors = {"Low": "#4CAF50", "Moderate": "#FF9800", "High": "#F44336"}
    color = level_colors.get(risk_level, "#666666")

    ax_summary.text(0.5, 0.85, "Risk Level", fontsize=14, ha="center",
                    transform=ax_summary.transAxes)
    ax_summary.text(0.5, 0.6, risk_level, fontsize=36, ha="center",
                    fontweight="bold", color=color, transform=ax_summary.transAxes)

    # Interpretation text
    interpretations = {
        "Low": "Movement patterns are within normal biomechanical ranges.\nContinue regular training and monitoring.",
        "Moderate": "Some biomechanical risk factors detected.\nConsider targeted neuromuscular training.",
        "High": "Significant biomechanical risk factors detected.\nRecommend clinical evaluation and intervention.",
    }
    ax_summary.text(
        0.5, 0.25, interpretations.get(risk_level, ""),
        fontsize=11, ha="center", va="center",
        transform=ax_summary.transAxes,
        style="italic", wrap=True,
    )

    # 3. Contributing factors (bottom left)
    ax_factors = fig.add_axes([0.08, 0.05, 0.4, 0.35])
    if contributing_factors:
        names = [f["feature"].replace("_", " ").title()[:30] for f in contributing_factors[:5]]
        values = [f["importance"] for f in contributing_factors[:5]]
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(names)))

        y_pos = range(len(names))
        ax_factors.barh(y_pos, values, color=colors)
        ax_factors.set_yticks(y_pos)
        ax_factors.set_yticklabels(names, fontsize=9)
        ax_factors.set_xlabel("Contribution Weight", fontsize=10)
        ax_factors.set_title("Top Contributing Factors", fontsize=12)
        ax_factors.grid(True, alpha=0.3, axis="x")
    else:
        ax_factors.text(0.5, 0.5, "No factor data available",
                       ha="center", va="center", transform=ax_factors.transAxes)
        ax_factors.axis("off")

    # 4. Joint angle radar chart (bottom right)
    ax_radar = fig.add_axes([0.55, 0.05, 0.4, 0.35], polar=True)
    if joint_angles:
        categories = list(joint_angles.keys())
        values = list(joint_angles.values())

        # Normalize to 0-1 range for display
        max_val = max(abs(v) for v in values) if values else 1
        values_norm = [v / max_val for v in values]
        values_norm.append(values_norm[0])  # Close the polygon

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles.append(angles[0])

        ax_radar.plot(angles, values_norm, "o-", linewidth=2, color="#2196F3")
        ax_radar.fill(angles, values_norm, alpha=0.2, color="#2196F3")
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=8)
        ax_radar.set_title("Joint Angle Profile", fontsize=12, pad=20)
    else:
        ax_radar.text(0, 0, "No angle data", ha="center", va="center")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
