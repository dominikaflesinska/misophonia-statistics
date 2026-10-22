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

    # Load dataset
    df = pd.read_csv(DATA_URL, sep="\t")

    print(f"Dataset shape: {df.shape}")

    # Variables used in the analysis
    measures = [
        "Neck and Lower 3rd Angle",
        "Subnasal Angle",
        "Convexity Angle",
        "Mandibular Protrussion",
    ]

    outcome = "Misophonia Severity"
    binary_outcome = "Misophonia"

    # Convert measurements and severity to numeric values
    for column in measures + [outcome]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Normalize the Yes/No outcome
    df[binary_outcome] = (
        df[binary_outcome]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------
    # 3. Spearman correlations
    # --------------------------------------------------

    results = []

    for column in measures:
        subset = df[[column, outcome]].dropna()

        if len(subset) < 3:
            continue

        rho, p_value = spearmanr(
            subset[column],
            subset[outcome],
        )

        results.append(
            {
                "measure": column,
                "n": len(subset),
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise ValueError(
            "No valid data available for the correlation analysis."
        )

    # Correct for multiple statistical tests
    results_df["p_holm"] = multipletests(
        results_df["p_value"],
        method="holm",
    )[1]

    # Save results, with the strongest absolute correlations first
    results_df = results_df.sort_values(
        "spearman_rho",
        key=lambda values: values.abs(),
        ascending=False,
    )

    results_df.to_csv(
        ROOT / "correlation_results.csv",
        index=False,
    )

    print("\nSpearman correlations:")
    print(results_df.to_string(index=False))

    # --------------------------------------------------
    # PLOT 1: Correlations with misophonia severity
    # --------------------------------------------------

    plot_df = results_df.sort_values("spearman_rho")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    sns.barplot(
        data=plot_df,
        x="spearman_rho",
        y="measure",
        color="steelblue",
        ax=ax,
    )

    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_title(
        "Anatomical Measurements and Misophonia Severity"
    )
    ax.set_xlabel("Spearman correlation (rho)")
    ax.set_ylabel("Anatomical measurement")

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "spearman_correlations.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------
    # PLOT 2: Neck angle by severity level
    # --------------------------------------------------

    feature = "Neck and Lower 3rd Angle"

    plot_data = df[[feature, outcome]].dropna().copy()

    # Keep only severity levels from 0 to 4
    plot_data = plot_data[
        plot_data[outcome].between(0, 4)
    ]

    plot_data[outcome] = plot_data[outcome].astype(int)

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.boxplot(
        data=plot_data,
        x=outcome,
        y=feature,
        order=[0, 1, 2, 3, 4],
        color="lightblue",
        showfliers=False,
        ax=ax,
    )

    sns.stripplot(
        data=plot_data,
        x=outcome,
        y=feature,
        order=[0, 1, 2, 3, 4],
        color="darkslategray",
        alpha=0.55,
        jitter=0.18,
        size=4,
        ax=ax,
    )

    ax.set_title(
        "Neck and Lower 3rd Angle by Misophonia Severity"
    )
    ax.set_xlabel("Misophonia Severity (0–4)")
    ax.set_ylabel("Neck and Lower 3rd Angle (degrees)")

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "angle_by_severity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------
    # PLOT 3: Misophonia prevalence by Convexity Angle
    # --------------------------------------------------

    feature = "Convexity Angle"

    prevalence_data = df[
        [feature, binary_outcome]
    ].copy()

    prevalence_data[feature] = pd.to_numeric(
        prevalence_data[feature],
        errors="coerce",
    )

    # Convert Yes/No into 1/0
    prevalence_data["Misophonia_yes"] = (
        prevalence_data[binary_outcome]
        .map({"yes": 1, "no": 0})
    )

    prevalence_data = prevalence_data.dropna(
        subset=[feature, "Misophonia_yes"]
    )

    # Divide Convexity Angle into five quantile groups
    prevalence_data["Angle group"] = pd.qcut(
        prevalence_data[feature],
        q=5,
        duplicates="drop",
    )

    # Calculate percentage with misophonia in each group
    prevalence_summary = (
        prevalence_data.groupby(
            "Angle group",
            observed=True,
        )
        .agg(
            Percentage=(
                "Misophonia_yes",
                lambda values: values.mean() * 100,
            ),
            Participants=("Misophonia_yes", "size"),
        )
        .reset_index()
    )

    # Save the summary table
    prevalence_summary.to_csv(
        ROOT / "convexity_group_summary.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=prevalence_summary,
        x="Angle group",
        y="Percentage",
        color="steelblue",
        ax=ax,
    )

    ax.set_title(
        "Misophonia Prevalence Across Convexity Angle Groups"
    )
    ax.set_xlabel(
        "Convexity Angle Group (lowest to highest)"
    )
    ax.set_ylabel("Participants with Misophonia (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "misophonia_by_convexity_group.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------
    # 4. Final output
    # --------------------------------------------------

    print("\nFiles saved successfully:")
    print("- correlation_results.csv")
    print("- convexity_group_summary.csv")
    print("- figures/spearman_correlations.png")
    print("- figures/angle_by_severity.png")
    print("- figures/misophonia_by_convexity_group.png")


if __name__ == "__main__":
    main()