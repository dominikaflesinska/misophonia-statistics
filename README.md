# Misophonia and Craniofacial Measurements

Statistical analysis of the association between craniofacial measurements and misophonia severity.

## Project overview

This project explores whether selected anatomical measurements are associated with self-reported misophonia severity. The analysis uses descriptive visualisation and Spearman rank correlations, with Holm correction for multiple testing.

The goal is to identify statistical associations, not to establish causal relationships.

## Dataset

- **Source:** [Misophonia dataset](https://alejandro-isglobal.github.io/SDA/Misophonia.txt)
- **Sample size:** 123 observations
- **Number of variables:** 21

The analysis focuses on four anatomical measurements and their associations with misophonia severity.

## Methods

- Data loading and numeric conversion
- Spearman rank correlation
- Holm correction for multiple comparisons
- Visualisation of correlation coefficients
- Boxplot and individual observations by severity level

## Key findings

All four anatomical measurements showed statistically significant Spearman correlations with misophonia severity after Holm correction.

| Anatomical measurement | Spearman's rho | Holm-adjusted p-value |
|---|---:|---:|
| Neck and Lower 3rd Angle | -0.424 | 0.000004 |
| Subnasal Angle | -0.398 | 0.000015 |
| Convexity Angle | -0.379 | 0.000030 |
| Mandibular Protrussion | 0.267 | 0.002849 |

The strongest association in absolute magnitude was observed for Neck and Lower 3rd Angle. The negative coefficients indicate that higher values of these measurements tended to be associated with lower misophonia severity scores in this dataset. Mandibular Protrussion showed a positive association.

These results describe associations and should not be interpreted as evidence of causation.

## Figures

### Spearman correlations

![Spearman correlation coefficients](figures/spearman_correlations.png)

### Neck and Lower 3rd Angle by misophonia severity

![Angle by severity](figures/angle_by_severity.png)

## How to run the analysis

1. Clone this repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt# misophonia-statistics

