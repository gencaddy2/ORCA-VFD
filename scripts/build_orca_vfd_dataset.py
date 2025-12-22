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


# Use paths relative to your project folder
RAW_DIR = Path("raw_drives")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PARQUET = OUTPUT_DIR / "orca_vfd_master.parquet"
SUMMARY_CSV = OUTPUT_DIR / "orca_vfd_file_summary.csv"


def parse_date_from_filename(filename: str):
    """
    Expect filenames like data-2-10-2022.csv
    Returns (year, month, day) as ints, plus a string date.
    """
    stem = Path(filename).stem  # data-2-10-2022
    parts = stem.split("-")     # ["data", "2", "10", "2022"]
    if len(parts) >= 4:
        month = int(parts[1])
        day = int(parts[2])
        year = int(parts[3])
        return year, month, day, f"{year:04d}-{month:02d}-{day:02d}"
    else:
        return None, None, None, None


def load_all_csvs(raw_dir: Path):
    all_frames = []
    summary_rows = []

    for csv_file in sorted(raw_dir.glob("*.csv")):
        print(f"Loading {csv_file.name} ...")
        df = pd.read_csv(csv_file)

        # Add metadata columns
        year, month, day, date_str = parse_date_from_filename(csv_file.name)
        df["source_file"] = csv_file.name
        df["log_date"] = date_str
        df["log_year"] = year
        df["log_month"] = month
        df["log_day"] = day

        # Normalize column names
        cols = (
            df.columns
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("%", "pct")
            .str.replace("(", "")
            .str.replace(")", "")
            .str.replace("/", "_")
            .str.replace("__", "_")
            .str.lower()
        )
        df.columns = cols

        all_frames.append(df)

        summary_rows.append(
            {
                "file": csv_file.name,
                "rows": len(df),
                "columns": df.shape[1],
                "log_date": date_str,
            }
        )

    if not all_frames:
        raise RuntimeError(f"No CSV files found in {raw_dir}")

    master_df = pd.concat(all_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    return master_df, summary_df


def main():
    print(f"Reading CSV files from: {RAW_DIR.resolve()}")
    master_df, summary_df = load_all_csvs(RAW_DIR)

    print()
    print("Summary of loaded data:")
    print(summary_df)

    total_rows = len(master_df)
    total_cols = master_df.shape[1]
    print(f"\nTotal rows: {total_rows}")
    print(f"Total columns per row: {total_cols}")
    print(f"Approx total data points: {total_rows * total_cols}")

    print(f"\nWriting master Parquet to: {MASTER_PARQUET}")
    master_df.to_parquet(MASTER_PARQUET, index=False)

    print(f"Writing per file summary CSV to: {SUMMARY_CSV}")
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
