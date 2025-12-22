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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


DATA_PATH = Path("output") / "orca_vfd_train_master.parquet"

# ORCA Core-8 candidates
CORE8 = [
    "orca_speed",
    "orca_torque",
    "orca_i_mag",
    "orca_v_dc",
    "orca_i_dc",
    "orca_p_out",
    "orca_temp_inv",
]


def ensure_fault_orca_features(df_fault: pd.DataFrame) -> pd.DataFrame:
    """
    For the Bacha fault dataset rows, derive ORCA Core-8 features
    from native columns like ia, ib, vdc, idc, power_dc, t1-3, etc.
    """
    df = df_fault.copy()

    # Current magnitude from ia, ib (we only have two phases)
    if {"ia", "ib"}.issubset(df.columns):
        df["orca_i_mag"] = np.sqrt(df["ia"] ** 2 + df["ib"] ** 2)
    else:
        df["orca_i_mag"] = np.nan

    # DC link voltage
    if "vdc" in df.columns:
        df["orca_v_dc"] = df["vdc"]
    else:
        df["orca_v_dc"] = np.nan

    # DC link current
    if "idc" in df.columns:
        df["orca_i_dc"] = df["idc"]
    else:
        df["orca_i_dc"] = np.nan

    # Output power – prefer DC power, fall back to AC power if needed
    power_dc = df["power_dc"] if "power_dc" in df.columns else pd.Series(np.nan, index=df.index)
    power_ac = df["power_ac"] if "power_ac" in df.columns else pd.Series(np.nan, index=df.index)
    df["orca_p_out"] = power_dc.fillna(power_ac)

    # Inverter temperature – mean of t1, t2, t3 if present
    if {"t1", "t2", "t3"}.issubset(df.columns):
        df["orca_temp_inv"] = (df["t1"] + df["t2"] + df["t3"]) / 3.0
    else:
        df["orca_temp_inv"] = np.nan

    # We do NOT have explicit speed or torque in Bacha – leave as NaN for now
    if "orca_speed" not in df.columns:
        df["orca_speed"] = np.nan
    if "orca_torque" not in df.columns:
        df["orca_torque"] = np.nan

    return df


def load_and_split(df):
    """Split into train and test using only Bacha fault data, with derived ORCA features."""

    # Keep only fault domain (Bacha)
    df_fault = df[df["orca_domain"] == "fault"].copy()

    # Derive ORCA features for fault dataset
    df_fault = ensure_fault_orca_features(df_fault)

    # Binary target: 1 = fault, 0 = normal (in Bacha, F0 is normal, others are faults)
    df_fault["fault_flag"] = np.where(df_fault["orca_fault_code"] > 0, 1, 0)

    # Decide which Core-8 features actually have some data in the fault domain
    available_features = []
    for feat in CORE8:
        if feat in df_fault.columns and not df_fault[feat].isna().all():
            available_features.append(feat)

    if not available_features:
        raise RuntimeError("No usable ORCA Core-8 features found in fault dataset.")

    print("\nUsable fault features:", available_features)

    X = df_fault[available_features]
    y = df_fault["fault_flag"]

    # Impute & scale
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, imputer, scaler, available_features


def train_model(X_train, y_train):
    """Train a RandomForest fault classifier."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, label="Test"):
    print(f"\n=== {label} Set (Bacha) Classification Report ===")
    print(classification_report(y_test, model.predict(X_test)))
    print(f"=== {label} Set Confusion Matrix ===")
    print(confusion_matrix(y_test, model.predict(X_test)))


def evaluate_other_domain(model, df, imputer, scaler, available_features, label):
    """Run inference on physics or field domain with same features."""
    df = df.copy()

    # binary label (all zeros for these domains, but we evaluate anyway)
    df["fault_flag"] = np.where(df["orca_fault_code"] > 0, 1, 0)

    # If some features are missing, fill with NaN
    for feat in available_features:
        if feat not in df.columns:
            df[feat] = np.nan

    X = df[available_features]
    y = df["fault_flag"]

    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    y_pred = model.predict(X_scaled)

    print(f"\n=== {label} Domain Fault Evaluation ===")
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))


def main():
    print("Loading unified dataset...")
    df = pd.read_parquet(DATA_PATH)

    # Train on Bacha fault domain
    X_train, X_test, y_train, y_test, imputer, scaler, available_features = load_and_split(df)
    print("Training samples:", len(X_train))

    model = train_model(X_train, y_train)

    # Evaluate on held-out Bacha fault data
    evaluate(model, X_test, y_test, label="Bacha Fault Dataset")

    # Evaluate on Field domain
    df_field = df[df["orca_domain"] == "field"]
    evaluate_other_domain(model, df_field, imputer, scaler, available_features, label="Field")

    # Evaluate on Physics domain
    df_phys = df[df["orca_domain"] == "physics"]
    evaluate_other_domain(model, df_phys, imputer, scaler, available_features, label="Physics")

    # Feature importances
    print("\n=== Feature Importances (Fault Binary) ===")
    for feat, imp in sorted(zip(available_features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"{feat}: {imp:.4f}")


if __name__ == "__main__":
    main()
