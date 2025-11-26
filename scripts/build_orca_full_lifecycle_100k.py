from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

BASE = Path(".")

FIELD_WITH_ANOM_PATH = BASE / "output" / "orca_vfd_field_with_anomaly.parquet"
PHYSICS_PATH = BASE / "physics_output" / "orca_vfd_physics_sample.csv"

OUT_PARQUET = BASE / "output" / "orca_vfd_full_lifecycle_100k.parquet"
OUT_CSV = BASE / "output" / "orca_vfd_full_lifecycle_100k.csv"

# ORCA Core-8
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

# Target life and placement
TOTAL_LIFE_HOURS = 100_000.0
FIELD_CENTER_HOURS = 90_000.0      # place real field data near end of life
INFANT_END_HOURS = 336.0           # ~2 weeks of operation


def to_float(x):
    return pd.to_numeric(x, errors="coerce")


def exp_model(t, a, b, c):
    t = np.asarray(t, dtype=float)
    return a * np.exp(b * t) + c


def fit_anomaly_curve(df_field: pd.DataFrame):
    """Fit anomaly(t) = a*exp(b t)+c on the raw field run_time_counter window."""
    df = df_field.dropna(subset=["run_time_counter", "anomaly_score"]).copy()
    df["run_time_counter"] = to_float(df["run_time_counter"])
    df["anomaly_score"] = to_float(df["anomaly_score"])

    t = df["run_time_counter"].values.astype(float)
    y = df["anomaly_score"].values.astype(float)

    p0 = [0.1, 0.01, y.mean()]
    params, _ = curve_fit(exp_model, t, y, p0=p0, maxfev=20000)
    return params  # a, b, c


def main():
    # ---------------------------
    # Load FIELD data
    # ---------------------------
    print(f"Loading field data from: {FIELD_WITH_ANOM_PATH}")
    df_field = pd.read_parquet(FIELD_WITH_ANOM_PATH)

    df_field["run_time_counter"] = to_float(df_field["run_time_counter"])
    df_field["anomaly_score"] = to_float(df_field["anomaly_score"])
    df_field = df_field.dropna(subset=["run_time_counter", "anomaly_score"])

    t_min = float(df_field["run_time_counter"].min())
    t_max = float(df_field["run_time_counter"].max())
    field_width = t_max - t_min

    print(f"Field runtime window (raw): {t_min} to {t_max} (width {field_width} hours)")

    # Fit drift model from field
    a, b, c = fit_anomaly_curve(df_field)
    print(f"Fitted anomaly(t) = {a} * exp({b} * t) + {c}")

    y_min = float(df_field["anomaly_score"].min())
    y_max = float(df_field["anomaly_score"].max())
    print(f"Field anomaly range: {y_min} to {y_max}")

    # ---------------------------
    # Load PHYSICS data
    # ---------------------------
    print(f"Loading physics data from: {PHYSICS_PATH}")
    df_phys = pd.read_csv(PHYSICS_PATH)

    phys_orca_cols = [c for c in df_phys.columns if c.startswith("orca_")]
    df_phys_core = df_phys[phys_orca_cols].copy()

    for col in CORE8:
        if col not in df_phys_core.columns:
            df_phys_core[col] = np.nan

    for col in df_phys_core.columns:
        df_phys_core[col] = to_float(df_phys_core[col])

    # ---------------------------
    # Map FIELD window to ~90,000 hours
    # ---------------------------
    field_start_global = FIELD_CENTER_HOURS - field_width / 2.0
    field_end_global = FIELD_CENTER_HOURS + field_width / 2.0

    print(f"Field mapped to global hours: {field_start_global} → {field_end_global}")

    df_mid = df_field.copy()
    # Align raw window [t_min, t_max] to [field_start_global, field_end_global]
    df_mid["life_time_hours"] = (df_mid["run_time_counter"] - t_min) + field_start_global
    df_mid["lifecycle_phase"] = "mid_field"

    for col in CORE8:
        if col not in df_mid.columns:
            df_mid[col] = np.nan
        df_mid[col] = to_float(df_mid[col])

    # ---------------------------
    # Phase 0 + 1: Infant + Useful Life
    # ---------------------------
    # We will build early life from 0 → field_start_global
    early_total_hours = field_start_global
    if early_total_hours <= INFANT_END_HOURS:
        raise RuntimeError("Field center is too early in life to separate infant/useful periods.")

    # Infant mortality region: 0 → INFANT_END_HOURS
    infant_n = 2000
    infant_hours = np.linspace(0.0, INFANT_END_HOURS, infant_n).astype(float)

    if len(df_phys_core) >= infant_n:
        df_infant = df_phys_core.sample(infant_n, random_state=42).copy()
    else:
        reps = int(np.ceil(infant_n / len(df_phys_core)))
        df_infant = pd.concat([df_phys_core] * reps, ignore_index=True).iloc[:infant_n].copy()

    df_infant["life_time_hours"] = infant_hours

    # Infant anomaly: mostly low, some spikes to capture early failures
    base_infant = y_min
    spread = y_max - y_min if y_max > y_min else 0.1
    noise = np.random.normal(scale=0.02 * spread, size=infant_n)
    anomaly_inf = base_infant + noise

    # Inject spikes in, say, 5 percent of samples
    spike_idx = np.random.choice(infant_n, size=int(0.05 * infant_n), replace=False)
    anomaly_inf[spike_idx] = base_infant + 0.7 * spread + np.random.normal(
        scale=0.1 * spread, size=len(spike_idx)
    )

    df_infant["anomaly_score"] = anomaly_inf
    df_infant["lifecycle_phase"] = "infant_mortality"

    # Useful life region: INFANT_END_HOURS → field_start_global
    useful_n = 8000
    useful_hours = np.linspace(INFANT_END_HOURS, early_total_hours, useful_n).astype(float)

    if len(df_phys_core) >= useful_n:
        df_useful = df_phys_core.sample(useful_n, random_state=43).copy()
    else:
        reps = int(np.ceil(useful_n / len(df_phys_core)))
        df_useful = pd.concat([df_phys_core] * reps, ignore_index=True).iloc[:useful_n].copy()

    df_useful["life_time_hours"] = useful_hours

    # Useful life anomaly: near low bound with tiny noise (stable operation)
    noise_use = np.random.normal(scale=0.01 * spread, size=useful_n)
    df_useful["anomaly_score"] = base_infant + noise_use
    df_useful["lifecycle_phase"] = "useful_life"

    # ---------------------------
    # Phase 3: Wearout (post-field → TOTAL_LIFE_HOURS)
    # ---------------------------
    late_n = 2000
    late_hours = np.linspace(field_end_global, TOTAL_LIFE_HOURS, late_n).astype(float)

    # Map late hours onto an "age" axis extending beyond the original field t_max
    # so the exponential model ramps up anomaly
    extra_age_span = field_width * 10.0
    t_late_age = np.linspace(t_max, t_max + extra_age_span, late_n).astype(float)
    anomaly_late = exp_model(t_late_age, a, b, c)

    df_late = pd.DataFrame({
        "life_time_hours": late_hours,
        "anomaly_score": anomaly_late,
        "lifecycle_phase": "wearout"
    })

    # Copy last known Core-8 values from the end of field as proxy for late-life signals
    last_mid = df_mid.sort_values("life_time_hours").iloc[-1]
    for col in CORE8:
        df_late[col] = float(last_mid[col]) if not pd.isna(last_mid[col]) else np.nan

    # ---------------------------
    # Combine all phases
    # ---------------------------
    keep_cols = ["life_time_hours", "lifecycle_phase", "anomaly_score"] + CORE8

    df_full = pd.concat(
        [df_infant[keep_cols], df_useful[keep_cols], df_mid[keep_cols], df_late[keep_cols]],
        ignore_index=True
    ).sort_values("life_time_hours").reset_index(drop=True)

    print("Full lifecycle shape:", df_full.shape)
    print(df_full["lifecycle_phase"].value_counts())

    # Save outputs
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(OUT_PARQUET, index=False)
    df_full.to_csv(OUT_CSV, index=False)

    print(f"Wrote: {OUT_PARQUET}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
