"""
04_regression.py
----------------
Panel fixed-effects regressions testing H1 and H2.

Input:  data/processed/panel_clean.parquet
Output: output/tables/regression_results.csv

Models
------
(1) Baseline:    ROA ~ DOI + controls                      (firm + year FE)
(2) H1 test:     ROA ~ DOI + DOI² + controls               (inverted U-shape)
(3) H2 test:     ROA ~ DOI + DOI² + R&D + DOI×R&D + controls (moderation)

Estimator: linearmodels PanelOLS with two-way fixed effects and firm-clustered SEs.

Reading results
---------------
H1 supported if: β(DOI) > 0  AND  β(DOI²) < 0
H2 supported if: β(DOI × R&D) > 0
Stars: *** p<0.01, ** p<0.05, * p<0.10
"""

import warnings
from pathlib import Path

import pandas as pd
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/processed/panel_clean.parquet")
TABLE_PATH = Path("output/tables")
TABLE_PATH.mkdir(parents=True, exist_ok=True)

# ── Load & Set Panel Index ────────────────────────────────────────────────────
# linearmodels requires a MultiIndex: (entity, time)
df = pd.read_parquet(DATA_PATH)
df = df.set_index(["gvkey", "fyear"])

print(f"Panel: {len(df):,} obs | {df.index.get_level_values('gvkey').nunique():,} firms")

CONTROLS = ["ln_at", "leverage", "age"]


# ── Helper: two-way FE regression ────────────────────────────────────────────
def run_fe(dep: str, indep: list[str]) -> object:
    """
    Estimate two-way (firm + year) fixed effects with firm-clustered SEs.

    Parameters
    ----------
    dep   : dependent variable name
    indep : list of independent variable names (controls added automatically)
    """
    formula_vars = indep + CONTROLS
    sub = df[[dep, *formula_vars]].dropna()
    formula = f"{dep} ~ {' + '.join(formula_vars)} + EntityEffects + TimeEffects"
    mod = PanelOLS.from_formula(formula, data=sub)
    return mod.fit(cov_type="clustered", cluster_entity=True)


# ── Estimate three models ─────────────────────────────────────────────────────
print("\nEstimating models...")
res1 = run_fe("roa", ["doi"])
print("  Model 1 (baseline) done")

res2 = run_fe("roa", ["doi", "doi_sq"])
print("  Model 2 (H1: non-linearity) done")

res3 = run_fe("roa", ["doi", "doi_sq", "rd_intensity", "doi_x_rd"])
print("  Model 3 (H2: moderation) done")


# ── Build Results Table ───────────────────────────────────────────────────────
KEY_VARS = ["doi", "doi_sq", "rd_intensity", "doi_x_rd"] + CONTROLS

model_labels = ["(1) Baseline", "(2) H1: Nonlinearity", "(3) H2: Moderation"]
models = [res1, res2, res3]

rows = []
for label, res in zip(model_labels, models):
    col: dict = {"Model": label}
    for var in KEY_VARS:
        if var in res.params.index:
            coef  = res.params[var]
            se    = res.std_errors[var]
            pval  = res.pvalues[var]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            col[var]          = f"{coef:.3f}{stars}"
            col[f"{var}_se"]  = f"({se:.3f})"
        else:
            col[var]         = ""
            col[f"{var}_se"] = ""
    col["N"]  = f"{int(res.nobs):,}"
    col["R²"] = f"{res.rsquared:.3f}"
    rows.append(col)

results_df = pd.DataFrame(rows).set_index("Model").T
print("\n=== Regression Results ===")
print(results_df.to_string())
results_df.to_csv(TABLE_PATH / "regression_results.csv")
print(f"\nSaved regression_results.csv")


# ── H1: Inflection Point ──────────────────────────────────────────────────────
print("\n--- H1 Diagnostic ---")
if "doi" in res2.params.index and "doi_sq" in res2.params.index:
    b1 = res2.params["doi"]
    b2 = res2.params["doi_sq"]
    if b2 < 0:
        inflection = -b1 / (2 * b2)
        sample_mean_doi = df["doi"].mean()
        print(f"  β(DOI)  = {b1:.3f}   β(DOI²) = {b2:.3f}")
        print(f"  → Inverted U-shape confirmed (β₂ < 0)")
        print(f"  → Performance-maximizing DOI = {inflection:.3f}")
        print(f"  → Sample mean DOI            = {sample_mean_doi:.3f}")
        if inflection > sample_mean_doi:
            print("  → Most firms are still on the upward slope of the curve")
        else:
            print("  → Most firms are past the performance-maximizing DOI threshold")
    else:
        print(f"  β(DOI²) = {b2:.3f} > 0 → U-shape (not inverted) → H1 not supported")

# ── H2: Moderation ────────────────────────────────────────────────────────────
print("\n--- H2 Diagnostic ---")
if "doi_x_rd" in res3.params.index:
    b_mod = res3.params["doi_x_rd"]
    p_mod = res3.pvalues["doi_x_rd"]
    stars = "***" if p_mod < 0.01 else "**" if p_mod < 0.05 else "*" if p_mod < 0.1 else "(n.s.)"
    print(f"  β(DOI × R&D) = {b_mod:.3f} {stars}  (p = {p_mod:.3f})")
    if b_mod > 0 and p_mod < 0.1:
        print("  → H2 supported: R&D intensity positively moderates DOI–performance")
    else:
        print("  → H2 not supported at conventional significance levels")

print("""
─────────────────────────────────────────────────────────────
Interpretation guide:
  Stars: *** p<0.01, ** p<0.05, * p<0.10
  SEs in parentheses, clustered at firm level
  All models: firm FE + year FE (two-way fixed effects)
─────────────────────────────────────────────────────────────
""")
