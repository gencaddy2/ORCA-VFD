from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np

# Path to unified ORCA-VFD training dataset
DATA_PATH = Path("output") / "orca_vfd_train_master.parquet"

# ORCA Core-8 feature names we would like to use
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


def main():
    print(f"Loading unified dataset from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    # 1. Decide which Core-8 features actually have *some* non-null values
    available_features = []
    for feat in CORE8:
        if feat in df.columns and not df[feat].isna().all():
            available_features.append(feat)

    print("\nRequested Core-8 features:", CORE8)
    print("Usable Core-8 features (non-all-NaN):", available_features)

    if not available_features:
        raise RuntimeError("No usable ORCA Core-8 features found in the dataset.")

    # 2. Build a working dataframe with only usable features + domain label
    cols_to_use = available_features + ["orca_domain"]
    df_core = df[cols_to_use].copy()

    # Drop rows where domain is missing (shouldn't happen, but safe)
    df_core = df_core.dropna(subset=["orca_domain"])

    # Encode domain as numeric label
    df_core["domain_label"] = df_core["orca_domain"].astype("category").cat.codes
    domain_mapping = dict(
        enumerate(df_core["orca_domain"].astype("category").cat.categories)
    )
    print("\nDomain mapping (label -> name):", domain_mapping)

    X = df_core[available_features]
    y = df_core["domain_label"]

    # 3. Impute missing values (mean per feature), then scale
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain size:", X_train.shape[0], "Test size:", X_test.shape[0])

    # 5. Random Forest baseline model
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining RandomForestClassifier...")
    model.fit(X_train, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test)

    print("\n=== Classification Report (Domain Classification) ===")
    print(classification_report(y_test, y_pred, target_names=[domain_mapping[i] for i in sorted(domain_mapping)]))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # 7. Feature importance
    importances = model.feature_importances_
    print("\n=== Feature Importances ===")
    for feat, imp in sorted(zip(available_features, importances), key=lambda x: -x[1]):
        print(f"{feat}: {imp:.4f}")


if __name__ == "__main__":
    main()
