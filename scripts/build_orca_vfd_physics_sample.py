from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(".")
RAW_DIR = BASE_DIR / "physics_raw"
OUT_DIR = BASE_DIR / "physics_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# TODO: change these to your actual filenames
PHYSICS_FILES = [
    RAW_DIR / "hanke_data_1.csv",
    RAW_DIR / "hanke_data_2.csv",
]

# How many rows from each file to sample for now
ROWS_PER_FILE = 500_000  # adjust as you like


def map_to_orca_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Hanke-style physics columns to ORCA-VFD Core-8.
    You will need to adjust the column names based on the actual dataset.
    Example assumes columns like:
      omega_m  -> mechanical speed (rad/s)
      tau_em   -> electromagnetic torque (Nm)
      ia, ib, ic -> phase currents
      u_dc     -> DC link voltage
      i_dc     -> DC link current (if present)
      p_out    -> output power (if present)
    """
    # Make a copy so we don't overwrite original columns
    df = df.copy()

    # Example mappings – CHANGE names to match your actual columns
    if "omega_m" in df.columns:
        df["orca_speed"] = df["omega_m"] * 60.0 / (2.0 * np.pi)  # rad/s → rpm
    else:
        df["orca_speed"] = np.nan

    if "tau_em" in df.columns:
        df["orca_torque"] = df["tau_em"]
    else:
        df["orca_torque"] = np.nan

    # Current magnitude from phase currents if available
    if {"ia", "ib", "ic"}.issubset(df.columns):
        df["orca_i_mag"] = np.sqrt(df["ia"]**2 + df["ib"]**2 + df["ic"]**2)
    else:
        df["orca_i_mag"] = np.nan

    # DC link voltage
    if "u_dc" in df.columns:
        df["orca_v_dc"] = df["u_dc"]
    else:
        df["orca_v_dc"] = np.nan

    # DC current (if present)
    if "i_dc" in df.columns:
        df["orca_i_dc"] = df["i_dc"]
    else:
        df["orca_i_dc"] = np.nan

    # Output power
    if "p_out" in df.columns:
        df["orca_p_out"] = df["p_out"]
    elif {"tau_em", "omega_m"}.issubset(df.columns):
        # P = torque * omega  (Nm * rad/s) → W
        df["orca_p_out"] = df["tau_em"] * df["omega_m"]
    else:
        df["orca_p_out"] = np.nan

    # Inverter temp: usually not in physics models
    df["orca_temp_inv"] = np.nan

    # Physics baseline: assume normal operation (no fault labels yet)
    df["orca_fault_code"] = 0

    return df


def main():
    all_frames = []

    for fpath in PHYSICS_FILES:
        print(f"Loading sample from {fpath.name} ...")
        # Sample the first N rows to keep it light for now
        df_sample = pd.read_csv(fpath, nrows=ROWS_PER_FILE)

        # Map to ORCA Core-8
        df_orca = map_to_orca_core(df_sample)
        df_orca["source_file"] = fpath.name

        all_frames.append(df_orca)

    physics_sample = pd.concat(all_frames, ignore_index=True)

    print("Physics sample shape:", physics_sample.shape)
    out_csv = OUT_DIR / "orca_vfd_physics_sample.csv"
    physics_sample.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()
