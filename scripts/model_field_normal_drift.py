from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


DATA_PATH = Path("output") / "orca_vfd_train_master.parquet"

CORE8 = [
    "orca_speed",
    "orca_torque",
    "orca_i_mag",
    "orca_v_dc",
    "orca_i_dc",
    "orca_p_out",
    "orca_temp_inv",
]


def main():
    print("Loading unified dataset...")
    df = pd.read_parquet(DATA_PATH)

    # Separate domains
    df_field = df[df["orca_domain"] == "field"].copy()
    df_phys = df[df["orca_domain"] == "physics"].copy()
    df_fault = df[df["orca_domain"] == "fault"].copy()

    print("Field shape:", df_field.shape)
    print("Physics shape:", df_phys.shape)
    print("Fault shape:", df_fault.shape)

    # Keep only Core-8 features that have some data in field
    available_features = []
    for feat in CORE8:
        if feat in df_field.columns and not df_field[feat].isna().all():
            available_features.append(feat)

    if not available_features:
        raise RuntimeError("No usable ORCA features in field dataset.")

    print("\nUsing features:", available_features)

    # Build training matrix from field only
    X_field = df_field[available_features]

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_field_imp = imputer.fit_transform(X_field)
    X_field_scaled = scaler.fit_transform(X_field_imp)

    # IsolationForest for normal behavior
    print("\nTraining IsolationForest on FIELD domain only...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.01,  # assume about 1 percent anomalies in field
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_field_scaled)

    # Compute anomaly scores for field
    # Note: IsolationForest gives higher scores for more normal points
    # We invert it so higher = more anomalous
    field_scores_raw = iso.decision_function(X_field_scaled)
    df_field["anomaly_score"] = -field_scores_raw

    print("\nField anomaly score summary:")
    print(df_field["anomaly_score"].describe())

    # Attach scores back to full dataframe for other domains
    def score_domain(df_dom, label):
        if df_dom.empty:
            return None

        X_dom = df_dom[available_features].copy()

        # ensure same columns
        for feat in available_features:
            if feat not in X_dom.columns:
                X_dom[feat] = np.nan

        X_dom_imp = imputer.transform(X_dom)
        X_dom_scaled = scaler.transform(X_dom_imp)

        scores_raw = iso.decision_function(X_dom_scaled)
        scores = -scores_raw

        df_dom = df_dom.copy()
        df_dom["anomaly_score"] = scores

        print(f"\n{label} anomaly score summary:")
        print(df_dom["anomaly_score"].describe())

        return df_dom

    df_phys_scored = score_domain(df_phys, "Physics")
    df_fault_scored = score_domain(df_fault, "Fault")

    # Show top 10 most anomalous FIELD samples
    print("\nTop 10 most anomalous FIELD samples:")
    cols_to_show = [
        "timestamp",
        "run_time_counter",
        "orca_speed",
        "orca_torque",
        "orca_i_mag",
        "orca_v_dc",
        "orca_i_dc",
        "orca_p_out",
        "orca_temp_inv",
        "anomaly_score",
    ]
    for c in cols_to_show:
        if c not in df_field.columns:
            df_field[c] = np.nan

    df_field_sorted = df_field.sort_values("anomaly_score", ascending=False)
    print(df_field_sorted[cols_to_show].head(10))

    # Optionally write out scored field dataset for later RUL work
    scored_field_path = Path("output") / "orca_vfd_field_with_anomaly.parquet"
    df_field.to_parquet(scored_field_path, index=False)
    print(f"\nWrote field dataset with anomaly scores to: {scored_field_path}")

    # If you also want to persist physics and fault scores you could save them too
    if df_phys_scored is not None:
        df_phys_scored.to_parquet(Path("output") / "orca_vfd_physics_with_anomaly.parquet", index=False)
    if df_fault_scored is not None:
        df_fault_scored.to_parquet(Path("output") / "orca_vfd_fault_with_anomaly.parquet", index=False)


if __name__ == "__main__":
    main()
