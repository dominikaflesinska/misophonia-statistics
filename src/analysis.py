from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

DATA_URL = "https://alejandro-isglobal.github.io/SDA/Misophonia.txt"

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Main analysis
# --------------------------------------------------

def main():
    sns.set_theme(style="whitegrid", context="notebook")

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.22,
    })

    # Load dataset
    df = pd.read_csv(DATA_URL, sep="\t")
    df.columns = df.columns.str.strip()

    print(f"Dataset shape: {df.shape}")

    measures = [
        "Neck and Lower 3rd Angle",
        "Subnasal Angle",
        "Convexity Angle",
        "Mandibular Protrussion",
    ]

    outcome = "Misophonia Severity"
    binary_outcome = "Misophonia"

    # Convert measurements and severity to numeric
    for column in measures + [outcome]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[binary_outcome] = (
        df[binary_outcome]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------
    # 3. Spearman correlations + Holm correction
    # --------------------------------------------------

    results = []

    for column in measures:
        subset = df[[column, outcome]].dropna()

        if len(subset) < 3:
            continue

        rho, p_value = spearmanr(subset[column], subset[outcome])

        results.append({
            "measure": column,
            "n": len(subset),
            "spearman_rho": rho,
            "p_value": p_value,
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise ValueError("No valid data available for correlation analysis.")

    results_df["p_holm"] = multipletests(
        results_df["p_value"],
        method="holm",
    )[1]

    results_df = results_df.sort_values(
        "spearman_rho",
        key=lambda values: values.abs(),
        ascending=False,
    )

    results_df.to_csv(ROOT / "correlation_results.csv", index=False)

    print("\nSpearman correlations:")
    print(results_df.to_string(index=False))

    # --------------------------------------------------
    # PLOT 1: Correlations with misophonia severity
    # --------------------------------------------------

    plot_df = results_df.sort_values("spearman_rho")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [
        "#C66A43" if value < 0 else "#2878A5"
        for value in plot_df["spearman_rho"]
    ]

    bars = ax.barh(
        plot_df["measure"],
        plot_df["spearman_rho"],
        color=colors,
        height=0.65,
    )

    ax.axvline(0, color="#333333", linewidth=1.1)

    # Leave space for coefficient labels on both sides
    ax.set_xlim(-0.55, 0.40)

    for bar in bars:
        value = bar.get_width()
        offset = 0.012

        ax.text(
            value + (offset if value >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(
        "Craniofacial Measurements and Misophonia Severity",
        loc="left",
        pad=12,
    )
    ax.set_xlabel("Spearman correlation (rho)")
    ax.set_ylabel("Anatomical measurement")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.22)

    fig.text(
        0.12, 0.025,
        "* Correlations are associations, not evidence of causation.",
        fontsize=9,
        color="#444444",
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(
        FIGURES_DIR / "spearman_correlations.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------
    # PLOT 2: Scatter plot with linear trend for the strongest association

    feature = "Neck and Lower 3rd Angle"
    plot_data = df[[feature, outcome]].copy()
    plot_data[feature] = pd.to_numeric(plot_data[feature], errors="coerce")
    plot_data[outcome] = pd.to_numeric(plot_data[outcome], errors="coerce")
    plot_data = plot_data.dropna()
    plot_data = plot_data[plot_data[outcome].between(0, 4)]

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.regplot(
        data=plot_data,
        x=feature,
        y=outcome,
        scatter_kws={"alpha": 0.60, "color": "#477B9E", "s": 42},
        line_kws={"color": "#C45A46", "linewidth": 2},
        ci=95,
        ax=ax,
    )

    ax.set_title(
        "Neck and Lower 3rd Angle vs Misophonia Severity",
        loc="left",
        pad=12,
    )
    ax.set_xlabel("Neck and Lower 3rd Angle (degrees)")
    ax.set_ylabel("Misophonia Severity (0–4)")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.grid(axis="y", alpha=0.22)

    fig.text(
        0.12, 0.025,
        "Each point represents one participant. Line and 95% CI show a "
        "linear trend; the primary association was assessed using Spearman correlation.",
        fontsize=8,
        color="#444444",
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(
        FIGURES_DIR / "angle_by_severity.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # PLOT 3: Misophonia prevalence by Convexity Angle
    # --------------------------------------------------

    feature = "Convexity Angle"

    prevalence_data = df[[feature, binary_outcome]].copy()
    prevalence_data[feature] = pd.to_numeric(
        prevalence_data[feature],
        errors="coerce",
    )

    prevalence_data["Misophonia_yes"] = (
        prevalence_data[binary_outcome].map({"yes": 1, "no": 0})
    )

    prevalence_data = prevalence_data.dropna(
        subset=[feature, "Misophonia_yes"]
    )

    prevalence_data["Angle group"] = pd.qcut(
        prevalence_data[feature],
        q=5,
        duplicates="drop",
    )

    prevalence_summary = (
        prevalence_data.groupby("Angle group", observed=True)
        .agg(
            Percentage=("Misophonia_yes", lambda values: values.mean() * 100),
            Participants=("Misophonia_yes", "size"),
        )
        .reset_index()
    )

    prevalence_summary.to_csv(
        ROOT / "convexity_group_summary.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(11, 6.5))

    positions = list(range(len(prevalence_summary)))

    bars = ax.bar(
        positions,
        prevalence_summary["Percentage"],
        color="#2878A5",
        width=0.68,
        edgecolor="white",
        linewidth=0.8,
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    labels = [
        f"Group {i + 1}\n"
        f"{interval.left:.2f} to {interval.right:.2f}\n"
        f"(n={int(n)})"
        for i, (interval, n) in enumerate(
            zip(
                prevalence_summary["Angle group"],
                prevalence_summary["Participants"],
            )
        )
    ]

    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))

    ax.set_title(
        "Misophonia Prevalence Across Convexity Angle Groups",
        loc="left",
        pad=12,
    )
    ax.set_xlabel("Convexity Angle group (lowest to highest)")
    ax.set_ylabel("Participants with Misophonia (%)")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.22)

    fig.text(
        0.12, 0.025,
        "Exploratory comparison: group percentages do not establish "
        "causation or a threshold.",
        fontsize=9,
        color="#444444",
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(
        FIGURES_DIR / "misophonia_by_convexity_group.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nFiles saved successfully:")
    print("- correlation_results.csv")
    print("- convexity_group_summary.csv")
    print("- figures/spearman_correlations.png")
    print("- figures/angle_by_severity.png")
    print("- figures/misophonia_by_convexity_group.png")


if __name__ == "__main__":
    main()
