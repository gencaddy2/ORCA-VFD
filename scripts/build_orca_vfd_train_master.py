from pathlib import Path
import pandas as pd

# Base project directory (current directory)
BASE = Path(".")

# Input dataset paths
FIELD_PATH = BASE / "output" / "orca_vfd_master.parquet"
FAULT_PATH = BASE / "output" / "orca_vfd_fault_master.csv"
PHYSICS_PATH = BASE / "physics_output" / "orca_vfd_physics_sample.csv"

# Output unified training dataset
OUT_PATH = BASE / "output" / "orca_vfd_train_master.parquet"


def main():
    # Load field dataset (ACS880)
    print("Loading Field dataset...")
    df_field = pd.read_parquet(FIELD_PATH)
    df_field["orca_domain"] = "field"

    # Load fault dataset (Bacha PMSM)
    print("Loading Fault dataset...")
    df_fault = pd.read_csv(FAULT_PATH)
    df_fault["orca_domain"] = "fault"

    # Load physics dataset (Hanke sample)
    print("Loading Physics dataset...")
    df_phys = pd.read_csv(PHYSICS_PATH)
    df_phys["orca_domain"] = "physics"

    # Ensure all have orca_fault_code
    if "orca_fault_code" not in df_field.columns:
        df_field["orca_fault_code"] = 0  # assume normal for field data

    if "orca_fault_code" not in df_phys.columns:
        df_phys["orca_fault_code"] = 0  # physics samples are baseline normal

    # Union of all columns across datasets
    all_cols = set(df_field.columns) | set(df_fault.columns) | set(df_phys.columns)
    ordered_cols = sorted(all_cols)

    def align(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the given dataframe has all columns in ordered_cols.
        Missing columns are filled with NA, then columns are reordered.
        """
        for c in ordered_cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df[ordered_cols]

    print("Aligning columns...")
    df_field2 = align(df_field)
    df_fault2 = align(df_fault)
    df_phys2 = align(df_phys)

    print("Concatenating...")
    df_train = pd.concat([df_field2, df_fault2, df_phys2], ignore_index=True)

    print("Final unified shape:", df_train.shape)
    print(df_train["orca_domain"].value_counts())

    # Make sure output directory exists
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Fix mixed-type timestamp for Parquet:
    # convert everything to string so Arrow doesn't choke on floats/NaNs
    if "timestamp" in df_train.columns:
        df_train["timestamp"] = df_train["timestamp"].astype(str)

    # Write to Parquet
    df_train.to_parquet(OUT_PATH, index=False)
    print("Wrote:", OUT_PATH)


if __name__ == "__main__":
    main()
