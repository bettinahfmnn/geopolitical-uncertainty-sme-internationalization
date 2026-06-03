"""
03_descriptives.py
------------------
Variable construction, winsorizing, summary statistics and figures.

Input:  data/processed/panel_clean.parquet
Output: data/processed/panel_with_vars.parquet
        output/tables/summary_statistics.csv
        output/figures/correlation_matrix.png
        output/figures/dv_distribution.png
        output/figures/main_relationship.png

Research design
---------------
Y:   Capital Intensity = capx / at
X:   GPR Index         = global geopolitical risk index (Caldara & Iacoviello 2022)
Mod: Firm Size         = log(at)
Int: gpr_x_size        = gpr * ln_at
Controls:
     leverage          = dltt / at
     roa               = nicon / at
     cash_ratio        = che / at

Usage
-----
    python code/03_descriptives.py
    task descriptives
"""

import os, sys, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ── Find project root ─────────────────────────────────────────────────────────
def find_env():
    current = Path(os.getcwd())
    for path in [current] + list(current.parents):
        if (path / ".env").exists():
            return path / ".env"
        try:
            for s in path.iterdir():
                if s.is_dir() and (s / ".env").exists():
                    return s / ".env"
        except PermissionError:
            continue
    raise FileNotFoundError("Could not find .env anywhere.")

project_root = find_env().parent
os.chdir(project_root)
print(f"Project root: {project_root}")

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.family": "sans-serif"})
WU_BLUE = "#002f5f"
WU_RED  = "#c8102e"

# ── Paths ─────────────────────────────────────────────────────────────────────
IN_PATH    = Path("data") / "processed" / "panel_clean.parquet"
OUT_PANEL  = Path("data") / "processed" / "panel_with_vars.parquet"
TABLE_PATH = Path("output") / "tables"
FIG_PATH   = Path("output") / "figures"
TABLE_PATH.mkdir(parents=True, exist_ok=True)
FIG_PATH.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("\nLoading clean panel...")
df = pd.read_parquet(IN_PATH)
print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── Data quality filters ──────────────────────────────────────────────────────
print("\nApplying data quality filters...")
n = len(df)
df = df[(df["at"] > 0.1) & (df["sale"] > 0) & (df["seq"] > 0)].copy()
df = df[df["at"] >= 1].copy()
print(f"  After at>0.1, sale>0, seq>0, at>=1: {len(df):,} (removed {n-len(df):,})")

# ── SME filter ────────────────────────────────────────────────────────────────
n = len(df)
sme_mask = (df["emp"] < 0.25) | (df["at"] <= 43)
df = df[sme_mask].copy()
print(f"  After SME filter (emp<250 OR at<=43m): {len(df):,} (removed {n-len(df):,})")

# ── GPR merge ─────────────────────────────────────────────────────────────────
print("\nMerging GPR index...")
gpr_raw = pd.read_excel("data/raw/data_gpr_export.xls")

# Keep only actual data rows (skip label rows at top)
gpr_raw["GPR_num"] = pd.to_numeric(gpr_raw["GPR"], errors="coerce")
gpr_raw = gpr_raw[gpr_raw["GPR_num"].notna()].copy()
gpr_raw["month"] = pd.to_datetime(gpr_raw["month"], errors="coerce")
gpr_raw = gpr_raw.dropna(subset=["month"]).copy()
gpr_raw = gpr_raw[gpr_raw["month"].dt.year >= 1985].copy()
gpr_raw["fyear"] = gpr_raw["month"].dt.year

# Annual average of global GPR
gpr_annual = gpr_raw.groupby("fyear")["GPR_num"].mean().reset_index()
gpr_annual.columns = ["fyear", "gpr"]

print(f"  GPR years available: {gpr_annual['fyear'].min()} – {gpr_annual['fyear'].max()}")
print(f"  GPR range: {gpr_annual['gpr'].min():.1f} – {gpr_annual['gpr'].max():.1f} (should be ~50–200)")

# Merge on fiscal year
df = df.merge(gpr_annual, on="fyear", how="left")
print(f"  GPR merged: {df['gpr'].notna().sum():,} non-missing values")

# ── Variable construction ─────────────────────────────────────────────────────
print("\nConstructing variables...")

# Dependent variable
df["capx_intensity_dv"] = df["capx"].fillna(0) / df["at"]

# Control: RoA
df["roa"] = df["nicon"].fillna(0) / df["at"]

# Firm size (must come before interaction)
df["ln_at"] = df["at"].apply(lambda x: math.log(x) if x > 0 else np.nan)

# Interaction: GPR x Firm Size
df["gpr_x_size"] = df["gpr"] * df["ln_at"]

# Controls
df["leverage"]   = df["dltt"].fillna(0) / df["at"]
df["cash_ratio"] = df["che"].fillna(0) / df["at"]

# ── Drop missing core variables ───────────────────────────────────────────────
CORE_VARS = ["capx_intensity_dv", "roa", "ln_at", "leverage", "gpr"]
n = len(df)
df = df.dropna(subset=CORE_VARS).copy()
print(f"  Dropped {n-len(df):,} rows with missing core vars")
print(f"  Working sample: {len(df):,} firm-years | {df['gvkey'].nunique():,} firms")

# ── Winsorize at 1%-99% ───────────────────────────────────────────────────────
def winsorize(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

print("\nWinsorizing at 1%-99%...")
for col in ["capx_intensity_dv", "roa", "leverage", "cash_ratio"]:
    df[col] = winsorize(df[col])
    print(f"  {col:<20} [{df[col].min():>8.4f}, {df[col].max():>8.4f}]")

# Recompute interaction after winsorizing
df["gpr_x_size"] = df["gpr"] * df["ln_at"]

# ── Minimum 3 observations per firm ──────────────────────────────────────────
obs   = df.groupby("gvkey")["fyear"].count()
valid = obs[obs >= 3].index
n = len(df)
df = df[df["gvkey"].isin(valid)].copy()
print(f"\nMin 3 obs: {n:,} -> {len(df):,} | {df['gvkey'].nunique():,} firms")
print(f"Firms with CAPX>0: {(df['capx_intensity_dv']>0).sum():,} "
      f"({(df['capx_intensity_dv']>0).mean()*100:.1f}%)")

# ── Summary statistics ────────────────────────────────────────────────────────
VAR_LABELS = {
    "capx_intensity_dv": "Capital Intensity (capx/at)",
    "roa":               "RoA (nicon/at)",
    "gpr":               "Geopolitical Risk Index",
    "ln_at":             "Firm Size (log assets)",
    "leverage":          "Leverage (dltt/at)",
    "cash_ratio":        "Cash Ratio (che/at)",
}

summary = (
    df[list(VAR_LABELS.keys())]
    .rename(columns=VAR_LABELS)
    .describe(percentiles=[0.25, 0.5, 0.75])
    .T[["count","mean","std","min","25%","50%","75%","max"]]
    .round(4)
)
print("\n=== Summary Statistics ===")
print(summary.to_string())
summary.to_csv(TABLE_PATH / "summary_statistics.csv")
print(f"\nSaved summary_statistics.csv")

# ── Correlation matrix ────────────────────────────────────────────────────────
corr = df[list(VAR_LABELS.keys())].rename(columns=VAR_LABELS).corr().round(2)
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix — Research Variables",
             fontsize=13, color=WU_BLUE)
fig.tight_layout()
fig.savefig(FIG_PATH / "correlation_matrix.png", dpi=150)
plt.close()
print("Saved correlation_matrix.png")

# ── DV distribution + median CAPX intensity by year ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["capx_intensity_dv"], bins=60, color=WU_BLUE, alpha=0.8, edgecolor="white")
axes[0].axvline(df["capx_intensity_dv"].mean(),   color=WU_RED,   lw=2,
                label=f"Mean   = {df['capx_intensity_dv'].mean():.3f}")
axes[0].axvline(df["capx_intensity_dv"].median(), color="orange", lw=2, ls="--",
                label=f"Median = {df['capx_intensity_dv'].median():.3f}")
axes[0].set_xlabel("Capital Intensity (capx/at)")
axes[0].set_title("Distribution of Capital Intensity", color=WU_BLUE)
axes[0].legend()

yearly = df.groupby("fyear")["capx_intensity_dv"].median()
axes[1].bar(yearly.index, yearly.values, color=WU_BLUE, alpha=0.8)
axes[1].axhline(0, color="black", lw=0.8, ls="--")
axes[1].set_xlabel("Fiscal Year")
axes[1].set_ylabel("Median Capital Intensity")
axes[1].set_title("Median Capital Intensity by Year", color=WU_BLUE)
fig.tight_layout()
fig.savefig(FIG_PATH / "dv_distribution.png", dpi=150)
plt.close()
print("Saved dv_distribution.png")

# ── Main relationship: GPR vs Capital Intensity ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
df_plot = df.reset_index(drop=True)

# Left: scatter + bin means
axes[0].scatter(df_plot["gpr"], df_plot["capx_intensity_dv"],
                alpha=0.1, s=8, color=WU_BLUE)
bins = pd.cut(df_plot["gpr"], bins=15)
bm   = df_plot.groupby(bins, observed=True)[["gpr","capx_intensity_dv"]].mean()
axes[0].plot(bm["gpr"], bm["capx_intensity_dv"],
             color=WU_RED, lw=2.5, label="Bin mean")
axes[0].axhline(0, color="gray", lw=0.8, ls="--")
axes[0].set_xlabel("Geopolitical Risk Index")
axes[0].set_ylabel("Capital Intensity (capx/at)")
axes[0].set_title("GPR vs. Capital Intensity", color=WU_BLUE)
axes[0].legend()

# Right: median Capital Intensity by GPR — Small vs Large firms
df_plot["size_group"] = np.where(df_plot["ln_at"] < df_plot["ln_at"].median(),
                                 "Small firms", "Large firms")
df_plot["gpr_bin"] = pd.cut(df_plot["gpr"], bins=10)
palette2 = {"Small firms": "#2166ac", "Large firms": WU_RED}
for label, group in df_plot.groupby("size_group", observed=True):
    g  = group.reset_index(drop=True)
    bm = g.groupby("gpr_bin", observed=True)[["gpr","capx_intensity_dv"]].median()
    axes[1].plot(bm["gpr"], bm["capx_intensity_dv"], lw=2,
                 label=label, color=palette2[label],
                 marker="o", markersize=5)
axes[1].axhline(0, color="gray", lw=0.8, ls="--")
axes[1].set_xlabel("Geopolitical Risk Index")
axes[1].set_ylabel("Median Capital Intensity")
axes[1].set_title("Capital Intensity by GPR:\nSmall vs Large Firms", color=WU_BLUE)
axes[1].legend()

fig.suptitle("Main Relationship: GPR -> Capital Intensity",
             fontsize=13, color=WU_BLUE, y=1.02)
fig.tight_layout()
fig.savefig(FIG_PATH / "main_relationship.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved main_relationship.png")

# ── Save panel with variables ─────────────────────────────────────────────────
df.to_parquet(OUT_PANEL, index=False)
print(f"\nSaved panel_with_vars.parquet: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("Next step: python code/04_regression.py")