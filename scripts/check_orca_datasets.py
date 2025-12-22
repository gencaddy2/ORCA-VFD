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

BASE = Path(".")

field_path = BASE / "output" / "orca_vfd_master.parquet"
fault_path = BASE / "output" / "orca_vfd_fault_master.csv"
physics_path = BASE / "physics_output" / "orca_vfd_physics_sample.csv"

def show_info(name, df):
    print(f"\n=== {name} ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(3))

def main():
    # Field dataset
    df_field = pd.read_parquet(field_path)
    show_info("FIELD (ACS880)", df_field)

    # Fault dataset
    df_fault = pd.read_csv(fault_path)
    show_info("FAULT (Bacha)", df_fault)

    # Physics dataset
    df_phys = pd.read_csv(physics_path)
    show_info("PHYSICS (Hanke sample)", df_phys)

if __name__ == "__main__":
    main()
