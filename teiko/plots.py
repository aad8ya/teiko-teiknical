"""Part 3 boxplot visualizations comparing responders vs non-responders per cell population."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import pandas as pd  # noqa: E402

from teiko.config import FIGURES_DIR, POPULATIONS, TABLES_DIR  # noqa: E402
from teiko.stats import load_frequency_with_metadata, part3_subset  # noqa: E402

_POP_LABELS = {
    "b_cell": "B Cell",
    "cd8_t_cell": "CD8 T Cell",
    "cd4_t_cell": "CD4 T Cell",
    "nk_cell": "NK Cell",
    "monocyte": "Monocyte",
}


def load_plot_data() -> pd.DataFrame:
    """Load the Part 3 subset with a response_label column for display."""
    df = load_frequency_with_metadata()
    df = part3_subset(df)
    df = df[["population", "percentage", "response", "time_from_treatment_start"]].copy()
    df["response_label"] = df["response"].map({"yes": "Responder", "no": "Non-responder"})
    return df


def _annotate_ax(ax: plt.Axes, row: pd.Series) -> None:
    """Add p-value annotations to a single axes."""
    mwu_p = row["mwu_p_bh"]
    t_p = row["t_p_bh"]
    sig = bool(row["significant_primary"])
    marker = "**" if sig else "ns"

    y_max = ax.get_ylim()[1]
    ax.text(
        0.5,
        0.97,
        f"MWU BH p = {mwu_p:.3g}\nt BH p = {t_p:.3g}  {marker}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
    )


def plot_per_population(stats_df: pd.DataFrame, plot_df: pd.DataFrame) -> None:
    """Save one PNG per population to outputs/figures/."""
    for pop in POPULATIONS:
        pop_data = plot_df[plot_df["population"] == pop]
        row = stats_df[stats_df["population"] == pop].iloc[0]

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.boxplot(
            data=pop_data,
            x="response_label",
            y="percentage",
            order=["Responder", "Non-responder"],
            ax=ax,
        )
        ax.set_title(_POP_LABELS.get(pop, pop))
        ax.set_xlabel("Response group")
        ax.set_ylabel("Relative frequency (%)")
        _annotate_ax(ax, row)

        fig.savefig(FIGURES_DIR / f"boxplot_{pop}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def plot_combined(stats_df: pd.DataFrame, plot_df: pd.DataFrame) -> None:
    """Save a single wide figure with all 5 populations side-by-side."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    for ax, pop in zip(axes, POPULATIONS):
        pop_data = plot_df[plot_df["population"] == pop]
        row = stats_df[stats_df["population"] == pop].iloc[0]

        sns.boxplot(
            data=pop_data,
            x="response_label",
            y="percentage",
            order=["Responder", "Non-responder"],
            ax=ax,
        )
        ax.set_title(_POP_LABELS.get(pop, pop))
        ax.set_xlabel("Response group")
        ax.set_ylabel("Relative frequency (%)")
        _annotate_ax(ax, row)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "boxplots_combined.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_per_timepoint(plot_df: pd.DataFrame) -> None:
    """Save a faceted figure showing all populations across timepoints."""
    g = sns.catplot(
        data=plot_df,
        kind="box",
        x="time_from_treatment_start",
        y="percentage",
        col="population",
        col_wrap=5,
        hue="response_label",
        order=[0, 7, 14],
        hue_order=["Responder", "Non-responder"],
        height=4,
        aspect=0.9,
    )
    g.set_axis_labels("Days from treatment start", "Relative frequency (%)")
    for ax, pop in zip(g.axes.flat, POPULATIONS):
        ax.set_title(_POP_LABELS.get(pop, pop))

    g.savefig(FIGURES_DIR / "boxplots_per_timepoint.png", dpi=100, bbox_inches="tight")
    plt.close("all")


def generate_all_plots() -> None:
    """Generate all Part 3 boxplot figures. stats_baseline.csv must exist first."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    stats_df = pd.read_csv(TABLES_DIR / "stats_baseline.csv")
    plot_df = load_plot_data()

    plot_per_population(stats_df, plot_df)
    plot_combined(stats_df, plot_df)
    plot_per_timepoint(plot_df)
