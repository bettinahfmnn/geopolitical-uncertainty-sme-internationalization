import numpy as np
import pandas as pd
from pathlib import Path
import glob

# ── Find most recent pull folder ──────────────────────────────────────────────
raw_folders = sorted(glob.glob("data/raw/2*"))
if not raw_folders:
    raise FileNotFoundError("No raw data found. Run 01_pull_data.py first.")
latest = Path(raw_folders[-1])
print(f"Using data from: {latest}")

OUT_PATH = Path("data/processed/panel_clean.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load all parquet files ────────────────────────────────────────────────────
print("Loading raw data...")
files = sorted(latest.glob("fyear_*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df.columns = [c.strip().lower() for c in df.columns]
print(f"  Raw observations: {len(df):,} | firms: {df['gvkey'].nunique():,}")

# ── Drop duplicates ───────────────────────────────────────────────────────────
df = df.drop_duplicates(subset=["gvkey", "fyear"]).copy()

# ── Convert numeric columns ───────────────────────────────────────────────────
for col in ["capx", "at", "emp", "nicon", "sale", "ebit", "ebitda",
            "dltt", "dlc", "seq", "lt", "che", "act", "lct",
            "ppent", "xrd", "dp", "wcap", "re", "oancf", "intan"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── SME Filter ────────────────────────────────────────────────────────────────
if "emp" in df.columns and "at" in df.columns:
    sme_mask = (df["emp"] < 0.25) | (df["at"] <= 43)
    n_before = len(df)
    df = df[sme_mask].copy()
    print(f"  After SME filter: {len(df):,} (removed {n_before - len(df):,})")

# ── Construct Variables ───────────────────────────────────────────────────────
df["ln_at"] = np.log(df["at"].replace(0, np.nan))
df["capx_intensity"] = df["capx"] / df["at"]
df["roa"] = df["nicon"] / df["at"] if "nicon" in df.columns else np.nan
df["leverage"] = (df["dltt"] + df["dlc"]) / df["seq"]
df["rd_intensity"] = df["xrd"].fillna(0) / df["at"]

# ── Winsorize ─────────────────────────────────────────────────────────────────
def winsorize(series, lower=0.01, upper=0.99):
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

for col in ["capx_intensity", "roa", "leverage", "rd_intensity"]:
    if col in df.columns:
        df[col] = winsorize(df[col])

# ── Sort ──────────────────────────────────────────────────────────────────────
df = df.sort_values(["gvkey", "fyear"]).reset_index(drop=True)
print(f"  Final: {len(df):,} obs | {df['gvkey'].nunique():,} firms")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_parquet(OUT_PATH, index=False)

# Clean log
log = Path("data/processed/clean_log.txt")
log.write_text(
    f"Rows: {len(df):,}\n"
    f"Firms: {df['gvkey'].nunique():,}\n"
    f"Years: {df['fyear'].min()}-{df['fyear'].max()}\n"
)
print(f"Saved to {OUT_PATH}")