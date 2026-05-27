"""
02_clean.py
-----------
Clean raw Compustat data, apply SME filter, construct variables.

Input:  data/raw/compustat_global_raw.parquet
Output: data/processed/panel_clean.parquet

Variable construction
---------------------
ROA            = ib / at           (return on assets; performance)
DOI            = pifo / sale       (foreign income share; degree of internationalization)
DOI²           = doi ** 2          (non-linearity test, H1)
R&D intensity  = xrd / at          (R&D expenditure / total assets; 0 if missing)
DOI × R&D      = doi * rd_intensity (moderation term, H2)
Firm size      = log(at)
Leverage       = dltt / at
Age            = fyear - inco

All continuous outcome variables are winsorized at 1%–99%.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths — always relative ────────────────────────────────────────────────────
RAW_PATH = Path("data/raw/compustat_global_raw.parquet")
OUT_PATH = Path("data/processed/panel_clean.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading raw data...")
df = pd.read_parquet(RAW_PATH)
n_raw = len(df)
print(f"  Raw observations: {n_raw:,} | firms: {df['gvkey'].nunique():,}")


# ── SME Filter ────────────────────────────────────────────────────────────────
# EU definition: < 250 employees OR total assets ≤ €43m
# emp in Compustat is in thousands → 0.25 = 250 employees
sme_mask = (df["emp"] < 0.25) | (df["at"] <= 43)
df = df[sme_mask].copy()
print(f"  After SME filter: {len(df):,} (removed {n_raw - len(df):,})")


# ── Construct Variables ───────────────────────────────────────────────────────
# Performance
df["roa"] = df["ib"] / df["at"]

# Degree of internationalization (DOI)
# pifo can be negative (foreign losses) — we winsorize below
df["doi"] = df["pifo"] / df["sale"]
df["doi_sq"] = df["doi"] ** 2

# R&D intensity — treat missing xrd as zero (firm did not report R&D expenditure)
df["rd_intensity"] = df["xrd"].fillna(0) / df["at"]

# Interaction term for H2
df["doi_x_rd"] = df["doi"] * df["rd_intensity"]

# Controls
df["ln_at"] = np.log(df["at"])
df["leverage"] = df["dltt"] / df["at"]
df["age"] = (df["fyear"] - df["inco"].fillna(df["fyear"] - 10)).clip(lower=0)


# ── Winsorize at 1%–99% ───────────────────────────────────────────────────────
def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip series at given quantiles."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


for col in ["roa", "doi", "rd_intensity", "leverage"]:
    df[col] = winsorize(df[col])

# Recompute derived variables after winsorizing inputs
df["doi_sq"] = df["doi"] ** 2
df["doi_x_rd"] = df["doi"] * df["rd_intensity"]


# ── Drop Observations with Missing Core Variables ─────────────────────────────
core_vars = ["roa", "doi", "doi_sq", "rd_intensity", "doi_x_rd", "ln_at", "leverage", "age"]
n_before = len(df)
df = df.dropna(subset=core_vars).copy()
print(f"  After dropping missing core vars: {len(df):,} (removed {n_before - len(df):,})")


# ── Require ≥ 3 Observations per Firm (balanced panel assumption) ─────────────
obs_per_firm = df.groupby("gvkey")["fyear"].count()
valid_firms = obs_per_firm[obs_per_firm >= 3].index
n_before = len(df)
df = df[df["gvkey"].isin(valid_firms)].copy()
print(f"  After min-obs filter (≥3 per firm): {len(df):,} (removed {n_before - len(df):,})")
print(f"  Final: {len(df):,} obs | {df['gvkey'].nunique():,} firms | {df['loc'].nunique()} countries")
print(f"  Years: {df['fyear'].min()}–{df['fyear'].max()}")


# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved cleaned panel to {OUT_PATH}")

