"""Build reproducible figures from the compact public evidence package.

The script never reads raw market data or exploratory outputs. Every plotted
number comes from a CSV committed under ``results/public/us300_release_v1`` so a
reviewer can audit the figure against the table that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = PROJECT_ROOT / "results" / "public" / "us300_release_v1"
FIGURE_DIR = EVIDENCE_DIR / "figures"

INK = "#172121"
MUTED = "#61706d"
RIDGE = "#d97745"
LASSO = "#176b65"
GRID = "#d8dfdc"
WARNING = "#9b3f35"
NOTICE_TEXT = "RESEARCH EVIDENCE | NOT LIVE OR EXECUTION-GRADE PERFORMANCE"


def configure_style() -> None:
    """Use one restrained visual language across every public chart."""

    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f4ed",
            "axes.facecolor": "#f7f4ed",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def add_evidence_status_notice(fig: plt.Figure, y: float = 0.01) -> None:
    """把证据状态写进图片本身，避免脱离 README 后被误读。"""

    fig.text(
        0.5,
        y,
        NOTICE_TEXT,
        ha="center",
        color=WARNING,
        fontsize=8.5,
        fontweight="bold",
    )


def save_walk_forward_ic(fold_metrics: pd.DataFrame) -> None:
    """Plot both IC definitions for every fold and model."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    colors = {"ridge": RIDGE, "lasso": LASSO}
    for ax, metric, title in [
        (axes[0], "pearson_ic_mean", "Pearson IC by validation fold"),
        (axes[1], "spearman_ic_mean", "Rank IC by validation fold"),
    ]:
        for model, model_df in fold_metrics.groupby("model"):
            ordered = model_df.sort_values("fold")
            ax.plot(
                ordered["fold"],
                ordered[metric],
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=colors.get(model, MUTED),
                label=model.title(),
            )
        ax.axhline(0, color=GRID, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Expanding-window fold")
        ax.set_ylabel("Mean daily cross-sectional IC")
        ax.set_xticks(sorted(fold_metrics["fold"].unique()))
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("US300 | 10-day target | all validation folds", fontsize=15, fontweight="bold")
    add_evidence_status_notice(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURE_DIR / "walk_forward_ic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_feature_inventory(feature_families: pd.DataFrame) -> None:
    """Show the exact canonical candidate-feature composition."""

    plot_df = feature_families[feature_families["family"] != "candidate_feature"].copy()
    labels = {
        "raw_feature": "Raw market",
        "fundamental_raw": "Fundamental",
        "base_feature": "Base technical",
        "advanced_feature": "Advanced technical",
        "context_feature": "Context",
        "alpha_feature": "Alpha191",
    }
    plot_df["label"] = plot_df["family"].map(labels).fillna(plot_df["family"])
    plot_df = plot_df.sort_values("feature_count")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.barh(plot_df["label"], plot_df["feature_count"], color=LASSO)
    ax.bar_label(bars, padding=4, color=INK)
    ax.set_title("Canonical candidate feature inventory")
    ax.set_xlabel("Feature count")
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_xlim(0, max(plot_df["feature_count"].max() * 1.18, 1))
    add_evidence_status_notice(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURE_DIR / "feature_family_inventory.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_portfolio_diagnostic(portfolio: pd.DataFrame) -> None:
    """Compare selected 20 bps portfolio rows without hiding short sample size."""

    plot_df = portfolio.copy()
    if "holding_clock" not in plot_df.columns:
        raise ValueError("Public portfolio evidence must record holding_clock.")
    invalid_clocks = sorted(set(plot_df["holding_clock"].dropna().astype(str)) - {"signal_horizon"})
    if invalid_clocks:
        raise ValueError(
            "Public figures only accept holding_clock=signal_horizon; "
            f"found {invalid_clocks}."
        )
    plot_df["label"] = plot_df.apply(
        lambda row: f"Top{int(row['top_k'])} | "
        + ("sector-neutral" if row["neutral_mode"] == "sector_neutral" else "unconstrained"),
        axis=1,
    )
    colors = [LASSO if mode == "sector_neutral" else RIDGE for mode in plot_df["neutral_mode"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    returns = axes[0].barh(plot_df["label"], plot_df["cumulative_return"] * 100, color=colors)
    axes[0].bar_label(returns, fmt="%.1f%%", padding=4)
    axes[0].set_title("Cumulative long-short return")
    axes[0].set_xlabel("Percent over observed return days")
    axes[0].grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)

    sharpes = axes[1].barh(plot_df["label"], plot_df["sharpe"], color=colors)
    axes[1].bar_label(sharpes, fmt="%.2f", padding=4)
    axes[1].set_title("Annualized Sharpe diagnostic")
    axes[1].set_xlabel("Sharpe ratio")
    axes[1].grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)

    daily_count = ",".join(str(int(value)) for value in sorted(plot_df["daily_count"].dropna().unique()))
    rebalance_count = ",".join(
        str(int(value)) for value in sorted(plot_df["rebalance_count"].dropna().unique())
    )
    fig.suptitle(
        f"US300 portfolio diagnostic | 20 bps | {daily_count} days | {rebalance_count} rebalances",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Signal-horizon clock; short OOS sample; no calibrated slippage, nonlinear impact, or short-locate model.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    add_evidence_status_notice(fig, y=-0.01)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURE_DIR / "portfolio_diagnostic_20bps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_portfolio_cost_sensitivity(portfolio_grid: pd.DataFrame) -> None:
    """Show how one fixed portfolio recipe changes as the friction assumption rises."""

    required = {
        "hold_days",
        "holding_clock",
        "top_k",
        "cost_bps",
        "neutral_mode",
        "portfolio_total_return",
        "portfolio_sharpe",
    }
    if not required.issubset(portfolio_grid.columns):
        return

    portfolio_grid = portfolio_grid[
        portfolio_grid["holding_clock"].astype(str) == "signal_horizon"
    ].copy()

    plot_df = portfolio_grid[
        (pd.to_numeric(portfolio_grid["hold_days"], errors="coerce") == 10)
        & (pd.to_numeric(portfolio_grid["top_k"], errors="coerce") == 20)
        & (portfolio_grid["neutral_mode"].astype(str) == "sector_neutral")
    ].copy()
    if "weight_mode" in plot_df.columns:
        plot_df = plot_df[plot_df["weight_mode"].astype(str) == "equal_weight"].copy()
    if plot_df.empty:
        return

    plot_df = (
        plot_df.sort_values("cost_bps")
        .drop_duplicates(subset=["cost_bps"], keep="first")
        .reset_index(drop=True)
    )
    costs = pd.to_numeric(plot_df["cost_bps"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(costs, plot_df["portfolio_total_return"] * 100, marker="o", color=LASSO, linewidth=2.2)
    axes[0].set_title("Cumulative return sensitivity")
    axes[0].set_xlabel("All-in friction assumption (bps)")
    axes[0].set_ylabel("Cumulative return (%)")
    axes[0].grid(color=GRID, linewidth=0.7, alpha=0.8)

    axes[1].plot(costs, plot_df["portfolio_sharpe"], marker="o", color=RIDGE, linewidth=2.2)
    axes[1].set_title("Sharpe sensitivity")
    axes[1].set_xlabel("All-in friction assumption (bps)")
    axes[1].set_ylabel("Daily mean/std Sharpe")
    axes[1].grid(color=GRID, linewidth=0.7, alpha=0.8)

    fig.suptitle("Sector-neutral Top20 | 10-day non-overlapping sleeves", fontsize=14, fontweight="bold")
    add_evidence_status_notice(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURE_DIR / "portfolio_cost_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global NOTICE_TEXT
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = EVIDENCE_DIR / "experiment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        public_status = str(manifest.get("public_status", "pre_release_unknown"))
        if public_status == "release_candidate_requires_review":
            NOTICE_TEXT = "RELEASE CANDIDATE | CI, METRIC RECONCILIATION, AND MANUAL REVIEW REQUIRED"
        else:
            NOTICE_TEXT = f"PRE-RELEASE | {public_status.upper()}"
    configure_style()
    fold_metrics = pd.read_csv(EVIDENCE_DIR / "walk_forward_fold_metrics.csv")
    feature_families = pd.read_csv(EVIDENCE_DIR / "feature_family_summary.csv")
    portfolio = pd.read_csv(EVIDENCE_DIR / "portfolio_cost_summary.csv")
    save_walk_forward_ic(fold_metrics)
    save_feature_inventory(feature_families)
    save_portfolio_diagnostic(portfolio)
    portfolio_grid_path = EVIDENCE_DIR / "portfolio_grid_summary.csv"
    if portfolio_grid_path.exists():
        save_portfolio_cost_sensitivity(pd.read_csv(portfolio_grid_path))
    print(f"Public figures written to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
