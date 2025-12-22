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

BASE = Path(".")

FIELD_PATH = BASE / "output" / "orca_vfd_master.parquet"
FIELD_OUT_PATH = BASE / "output" / "orca_vfd_master.parquet"  # overwrite in place
FIELD_BACKUP_PATH = BASE / "output" / "orca_vfd_master_backup.parquet"


def main():
    print(f"Loading field dataset from: {FIELD_PATH}")
    df = pd.read_parquet(FIELD_PATH)
    print("Original shape:", df.shape)

    # Backup the original file once, just in case
    if not FIELD_BACKUP_PATH.exists():
        print(f"Creating backup at: {FIELD_BACKUP_PATH}")
        df.to_parquet(FIELD_BACKUP_PATH, index=False)

    cols = df.columns.tolist()
    print("\nAvailable columns in FIELD dataset:")
    for c in cols:
        print(" -", c)

    # --- ORCA Core-8 mappings for FIELD dataset ---

    # 1) orca_speed: use motor_speed_used if present
    if "motor_speed_used" in df.columns:
        df["orca_speed"] = df["motor_speed_used"]
    else:
        df["orca_speed"] = np.nan

    # 2) orca_torque: use motor_torque if present
    if "motor_torque" in df.columns:
        df["orca_torque"] = df["motor_torque"]
    else:
        df["orca_torque"] = np.nan

    # 3) orca_i_mag: use measured_absolute_current if present
    if "measured_absolute_current" in df.columns:
        df["orca_i_mag"] = df["measured_absolute_current"]
    else:
        df["orca_i_mag"] = np.nan

    # 4) orca_v_dc: use measured_dc_link_voltage if present
    if "measured_dc_link_voltage" in df.columns:
        df["orca_v_dc"] = df["measured_dc_link_voltage"]
    else:
        df["orca_v_dc"] = np.nan

    # 5) orca_p_out: prefer estimated_mechanical_power_at_motor_shaft, fallback to output_power
    mech_col = "estimated_mechanical_power_at_motor_shaft"
    if mech_col in df.columns:
        mech_power = df[mech_col]
    else:
        mech_power = pd.Series(np.nan, index=df.index)

    if "output_power" in df.columns:
        out_power = df["output_power"]
    else:
        out_power = pd.Series(np.nan, index=df.index)

    df["orca_p_out"] = mech_power.fillna(out_power)

    # 6) orca_i_dc: rough DC current estimate from power / Vdc
    #    I_dc ≈ P_out / V_dc (ignoring efficiency & wave-shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["orca_i_dc"] = df["orca_p_out"] / df["orca_v_dc"]
        df.loc[~np.isfinite(df["orca_i_dc"]), "orca_i_dc"] = np.nan

    # 7) orca_temp_inv: use inverter_temperatur if present
    if "inverter_temperatur" in df.columns:
        df["orca_temp_inv"] = df["inverter_temperatur"]
    else:
        df["orca_temp_inv"] = np.nan

    # 8) orca_fault_code: field data has no explicit faults here, set to 0
    df["orca_fault_code"] = 0

    print("\nAdded ORCA Core-8 columns:")
    core8 = [
        "orca_speed",
        "orca_torque",
        "orca_i_mag",
        "orca_v_dc",
        "orca_i_dc",
        "orca_p_out",
        "orca_temp_inv",
        "orca_fault_code",
    ]
    for c in core8:
        print(f" - {c}: non-null = {df[c].notna().sum()}")

    print("\nNew FIELD shape:", df.shape)

    # Overwrite the original FIELD file with ORCA Core-8 added
    FIELD_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FIELD_OUT_PATH, index=False)
    print(f"\nWrote updated field dataset with ORCA Core-8 to: {FIELD_OUT_PATH}")


if __name__ == "__main__":
    main()
