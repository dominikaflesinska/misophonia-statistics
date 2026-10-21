from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

DATA_URL = "https://alejandro-isglobal.github.io/SDA/Misophonia.txt"
ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    # Load the dataset
    df = pd.read_csv(DATA_URL, sep="\t")
    print(f"Dataset shape: {df.shape}")

    # Convert relevant columns to numeric values
    measures = [
        "Neck and Lower 3rd Angle",
        "Subnasal Angle",
        "Convexity Angle",
        "Mandibular Protrussion",
    ]
    outcome = "Misophonia Severity"

    for col in measures + [outcome]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Spearman correlations with misophonia severity
    results = []
    for col in measures:
        subset = df[[col, outcome]].dropna()
        rho, p_value = spearmanr(subset[col], subset[outcome])
        results.append({
            "measure": col,
            "n": len(subset),
            "spearman_rho": rho,
            "p_value": p_value,
        })

    results_df = pd.DataFrame(results)
    results_df["p_holm"] = multipletests(
        results_df["p_value"], method="holm"
    )[1]
    results_df = results_df.sort_values(
        "spearman_rho", key=lambda s: s.abs(), ascending=False
    )
    results_df.to_csv(ROOT / "correlation_results.csv", index=False)

    print("\nSpearman correlations:")
    print(results_df.to_string(index=False))

    # Plot 1: Correlations with severity
    sns.set_theme(style="whitegrid", context="notebook")
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
    ax.set_title("Anatomical Measurements and Misophonia Severity")
    ax.set_xlabel("Spearman correlation (rho)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "spearman_correlations.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Plot 2: Angle by severity
    feature = "Neck and Lower 3rd Angle"
    plot_data = df[[feature, outcome]].dropna().copy()
    plot_data[outcome] = plot_data[outcome].astype(int)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=plot_data,
        x=outcome,
        y=feature,
        color="lightblue",
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_data,
        x=outcome,
        y=feature,
        color="darkslategray",
        alpha=0.55,
        jitter=0.18,
        size=4,
        ax=ax,
    )
    ax.set_title("Neck and Lower 3rd Angle by Misophonia Severity")
    ax.set_xlabel("Misophonia Severity (0–4)")
    ax.set_ylabel("Neck and Lower 3rd Angle (degrees)")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "angle_by_severity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nPlots saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()