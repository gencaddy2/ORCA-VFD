"""
ORCA-VFD Research Code
Author: Carl L. Tolbert
Focus: Control-aware VFD sizing, torque-centric analysis, and reliability engineering
Context: Experimental and analytical research
Status: Non-safety-rated. Use for study, testing, and interpretation only.
License: MIT

Principle: Limits should be reached intentionally, not accidentally.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

BASE = Path(".")
FIELD_WITH_ANOM_PATH = BASE / "output" / "orca_vfd_field_with_anomaly.parquet"
PHYSICS_PATH = BASE / "physics_output" / "orca_vfd_physics_sample.csv"

OUT_PARQUET = BASE / "output" / "orca_vfd_full_lifecycle.parquet"
OUT_CSV = BASE / "output" / "orca_vfd_full_lifecycle.csv"

# We will use the ORCA Core-8
CORE8 = [
    "orca_speed",
    "orca_torque",
    "orca_i_mag",
    "orca_v_dc",
    "orca_i_dc",
    "orca_p_out",
    "orca_temp_inv",
    "orca_fault_code",
]


def to_float(x):
    """Safe float conversion."""
    return pd.to_numeric(x, errors="coerce")


def exp_model(t, a, b, c):
    """Exponential drift model."""
    t = np.asarray(t, dtype=float)   # <-- CRITICAL FIX
    return a * np.exp(b * t) + c


def fit_anomaly_curve(df_field):
    """Fit anomaly drift model."""
    df = df_field.dropna(subset=["run_time_counter", "anomaly_score"]).copy()

    df["run_time_counter"] = to_float(df["run_time_counter"])
    df["anomaly_score"] = to_float(df["anomaly_score"])

    t = df["run_time_counter"].values.astype(float)
    y = df["anomaly_score"].values.astype(float)

    p0 = [0.1, 0.01, y.mean()]
    params, _ = curve_fit(exp_model, t, y, p0=p0, maxfev=20000)
    return params


def build_full_lifecycle():
    # ------------------------------
    # Load field data
    # ------------------------------
    print(f"Loading field data from: {FIELD_WITH_ANOM_PATH}")
    df_field = pd.read_parquet(FIELD_WITH_ANOM_PATH)

    df_field["run_time_counter"] = to_float(df_field["run_time_counter"])
    df_field["anomaly_score"] = to_float(df_field["anomaly_score"])

    df_field = df_field.dropna(subset=["run_time_counter", "anomaly_score"])

    # Fit the exponential drift model
    print("Fitting exponential drift model...")
    a, b, c = fit_anomaly_curve(df_field)
    print(f"Model: anomaly(t) = {a} * exp({b} * t) + {c}")

    t_min = float(df_field["run_time_counter"].min())
    t_max = float(df_field["run_time_counter"].max())
    y_min = float(df_field["anomaly_score"].min())
    y_max = float(df_field["anomaly_score"].max())

    print(f"Field runtime range: {t_min} to {t_max}")
    print(f"Field anomaly range: {y_min} to {y_max}")

    # ------------------------------
    # Load physics dataset
    # ------------------------------
    print(f"Loading physics data from: {PHYSICS_PATH}")
    df_phys = pd.read_csv(PHYSICS_PATH)

    # Keep ORCA columns only
    cols_phys = [c for c in df_phys.columns if c.startswith("orca_")]
    df_phys_core = df_phys[cols_phys].copy()

    # Ensure all Core8 exist
    for col in CORE8:
        if col not in df_phys_core:
            df_phys_core[col] = np.nan

    # Convert physics values to float
    for col in df_phys_core.columns:
        df_phys_core[col] = to_float(df_phys_core[col])

    # ------------------------------
    # Phase 1: Early life (physics)
    # ------------------------------
    early_n = 5000
    if len(df_phys_core) > early_n:
        df_early = df_phys_core.sample(early_n, random_state=42).copy()
    else:
        df_early = df_phys_core.copy()

    # Early life duration
    early_hours = max(t_min, 50.0)
    df_early["life_time_hours"] = np.linspace(0, early_hours, len(df_early)).astype(float)

    # Synthetic anomaly (clamped)
    y_early = exp_model(df_early["life_time_hours"].values, a, b, c)
    df_early["anomaly_score"] = np.minimum(y_early, y_min)

    df_early["lifecycle_phase"] = "early_physics"

    # ------------------------------
    # Phase 2: Mid life (field)
    # ------------------------------
    df_mid = df_field.copy()
    df_mid["life_time_hours"] = df_mid["run_time_counter"] + early_hours
    df_mid["lifecycle_phase"] = "mid_field"

    for col in CORE8:
        if col not in df_mid.columns:
            df_mid[col] = np.nan
        df_mid[col] = to_float(df_mid[col])

    # ------------------------------
    # Phase 3: Late projected end-of-life
    # ------------------------------
    extension_hours = 200.0
    late_start = df_mid["life_time_hours"].max()
    late_end = late_start + extension_hours

    print(f"Projecting late life: {late_start} → {late_end} hours")

    t_late = np.linspace(late_start, late_end, 200).astype(float)
    anomaly_late = exp_model(t_late - early_hours, a, b, c)

    df_late = pd.DataFrame({
        "life_time_hours": t_late,
        "anomaly_score": anomaly_late,
        "lifecycle_phase": "late_projected"
    })

    # Fill Core8 using last field sample
    last_row = df_mid.sort_values("life_time_hours").iloc[-1]
    for col in CORE8:
        df_late[col] = float(last_row[col]) if not pd.isna(last_row[col]) else np.nan

    # ------------------------------
    # Combine all phases
    # ------------------------------
    all_cols = ["life_time_hours", "lifecycle_phase", "anomaly_score"] + CORE8
    df_full = pd.concat([
        df_early[all_cols],
        df_mid[all_cols],
        df_late[all_cols]
    ], ignore_index=True)

    df_full = df_full.sort_values("life_time_hours").reset_index(drop=True)

    print("Full lifecycle shape:", df_full.shape)
    print(df_full["lifecycle_phase"].value_counts())

    # Save outputs
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(OUT_PARQUET, index=False)
    df_full.to_csv(OUT_CSV, index=False)

    print(f"Wrote: {OUT_PARQUET}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    build_full_lifecycle()
