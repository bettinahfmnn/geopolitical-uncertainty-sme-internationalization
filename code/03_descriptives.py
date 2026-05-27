"""
03_descriptives.py
------------------
Summary statistics and exploratory figures.

Input:  data/processed/panel_clean.parquet
Output: output/tables/summary_statistics.csv
        output/figures/correlation_matrix.png
        output/figures/doi_roa_relationship.png
        output/figures/sample_composition.png

Notes on pandas index alignment
--------------------------------
When subsetting a DataFrame and then assigning new columns, always reset
the index to avoid the silent misalignment bug:

    high = df[df["sales"] > 400].copy()
    high.reset_index(drop=True, inplace=True)  # ← always do this
    high["score"] = pd.Series([10, 20])        # now aligns correctly
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.family": "sans-serif"})
WU_BLUE = "#002f5f"
WU_RED  = "#c8102e"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/processed/panel_clean.parquet")
TABLE_PATH  = Path("output/tables")
FIGURE_PATH = Path("output/figures")
TABLE_PATH.mkdir(parents=True, exist_ok=True)
FIGURE_PATH.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} observations | {df['gvkey'].nunique():,} firms")

# ── 1. Summary Statistics ─────────────────────────────────────────────────────
VAR_LABELS = {
    "roa":          "ROA",
    "doi":          "DOI (foreign income share)",
    "rd_intensity": "R&D intensity",
    "ln_at":        "Firm size (log assets)",
    "leverage":     "Leverage",
    "age":          "Firm age (years)",
}

summary = (
    df[list(VAR_LABELS.keys())]
    .rename(columns=VAR_LABELS)
    .describe(percentiles=[0.25, 0.5, 0.75])
    .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    .round(3)
)
print("\n=== Summary Statistics ===")
print(summary.to_string())
summary.to_csv(TABLE_PATH / "summary_statistics.csv")
print(f"Saved summary_statistics.csv")

# ── 2. Correlation Matrix ─────────────────────────────────────────────────────
corr_vars = list(VAR_LABELS.keys())
corr = df[corr_vars].rename(columns=VAR_LABELS).corr().round(2)

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
)
ax.set_title("Correlation Matrix — Key Variables", fontsize=13, pad=12, color=WU_BLUE)
fig.tight_layout()
fig.savefig(FIGURE_PATH / "correlation_matrix.png", dpi=150)
plt.close()
print("Saved correlation_matrix.png")

# ── 3. DOI–ROA Relationship (H1 preview) ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: raw scatter + bin means
axes[0].scatter(df["doi"], df["roa"], alpha=0.04, s=5, color=WU_BLUE)
bins = pd.cut(df["doi"], bins=20)
bin_means = df.groupby(bins, observed=True)[["doi", "roa"]].mean()
axes[0].plot(bin_means["doi"], bin_means["roa"], color=WU_RED, lw=2.5, label="Bin mean")
axes[0].set_xlabel("DOI (foreign income share)")
axes[0].set_ylabel("ROA")
axes[0].set_title("DOI vs. ROA — Raw Relationship", color=WU_BLUE)
axes[0].legend()

# Right: by R&D tercile (H2 preview)
# Reset index before assigning the tercile column — avoids pandas alignment bug
df_plot = df.copy()
df_plot.reset_index(drop=True, inplace=True)
df_plot["rd_tercile"] = pd.qcut(
    df_plot["rd_intensity"], q=3, labels=["Low R&D", "Mid R&D", "High R&D"]
)

palette = {"Low R&D": "#2166ac", "Mid R&D": "#f4a582", "High R&D": WU_RED}
for label, group in df_plot.groupby("rd_tercile", observed=True):
    group_reset = group.reset_index(drop=True)
    bins_g = pd.cut(group_reset["doi"], bins=15)
    bm = group_reset.groupby(bins_g, observed=True)[["doi", "roa"]].mean()
    axes[1].plot(bm["doi"], bm["roa"], lw=2, label=label, color=palette[label])

axes[1].set_xlabel("DOI (foreign income share)")
axes[1].set_ylabel("ROA")
axes[1].set_title("DOI vs. ROA by R&D Tercile (H2 preview)", color=WU_BLUE)
axes[1].legend()

fig.suptitle(
    "Degree of Internationalization & Firm Performance — European SMEs",
    fontsize=13, y=1.02, color=WU_BLUE,
)
fig.tight_layout()
fig.savefig(FIGURE_PATH / "doi_roa_relationship.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved doi_roa_relationship.png")

# ── 4. Sample Composition ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

country_counts = df["loc"].value_counts().head(10)
axes[0].barh(country_counts.index[::-1], country_counts.values[::-1], color=WU_BLUE)
axes[0].set_xlabel("Firm-year observations")
axes[0].set_title("Top 10 Countries in Sample", color=WU_BLUE)

year_counts = df["fyear"].value_counts().sort_index()
axes[1].bar(year_counts.index, year_counts.values, color=WU_BLUE)
axes[1].set_xlabel("Fiscal Year")
axes[1].set_ylabel("Observations")
axes[1].set_title("Sample Coverage by Year", color=WU_BLUE)

fig.tight_layout()
fig.savefig(FIGURE_PATH / "sample_composition.png", dpi=150)
plt.close()
print("Saved sample_composition.png")

print("\nDescriptives complete. Check output/tables/ and output/figures/")
