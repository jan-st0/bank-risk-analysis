import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from xgboost import XGBClassifier
from domain.model.portfolio_containers import TrainingBatch, npFloat64
from ports.ml_port import ModelTrainingPort, ProbabilityOfDefaultInferencePort

class VectorizedCalibratedXGBoostAdapter(ModelTrainingPort, ProbabilityOfDefaultInferencePort):
    def __init__(self, model_file_name: str = "calibrated_pd_xgboost.pkl"):
        self._model_file_name = model_file_name
        self._calibrator: CalibratedClassifierCV | None = None
        self._categorical_features = ["property_type", "occupancy_status"]
        self._numeric_features = ["current_actual_upb", "borrower_credit_score", "ltv_ratio", "debt_to_income", "loan_age"]

    def transform_raw_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes zero-copy string decoding and formatting using vectorized operations.
        Replaces individual lambda transformations with optimized element-level conversions.
        """
        working_df = df.copy()
        for col in self._categorical_features:
            if col in working_df.columns:
                if working_df[col].dtype == object or isinstance(working_df[col].dtype, pd.CategoricalDtype):
                    sample = working_df[col].iloc[0] if len(working_df) > 0 else None
                    if isinstance(sample, bytes):
                        working_df[col] = working_df[col].str.decode("utf-8")
                working_df[col] = working_df[col].astype(str).str.strip()
        
        for col in self._numeric_features:
            if col in working_df.columns:
                working_df[col] = pd.to_numeric(working_df[col], errors="coerce").fillna(0.0)
        return working_df

    def train_and_calibrate(self, batch: TrainingBatch, save_dir: str) -> dict[str, float]:
        X = pd.DataFrame(batch.features, columns=batch.feature_names)
        y = batch.targets

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self._numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self._categorical_features),
            ],
            remainder="drop"
        )

        base_xgb = XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )

        core_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", base_xgb)
        ])

        self._calibrator = CalibratedClassifierCV(
            estimator=core_pipeline,
            method="isotonic",
            cv=5
        )

        self._calibrator.fit(X_train, y_train)
        
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self._calibrator, os.path.join(save_dir, self._model_file_name))

        val_probs = self._calibrator.predict_proba(X_val)[:, 1]
        return {
            "roc_auc": float(roc_auc_score(y_val, val_probs)),
            "brier_score": float(brier_score_loss(y_val, val_probs)),
            "log_loss": float(log_loss(y_val, val_probs))
        }

    def load_model_asset(self, save_dir: str) -> None:
        target_path = os.path.join(save_dir, self._model_file_name)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"The required serialization asset could not be found at: {target_path}")
        self._calibrator = joblib.load(target_path)

    def predict_probabilities_direct(self, underlying_features: pd.DataFrame) -> npFloat64:
        if self._calibrator is None:
            raise RuntimeError("The internal scoring pipeline is uninitialized.")
        processed_df = self.transform_raw_dataframe(underlying_features)
        return self._calibrator.predict_proba(processed_df)[:, 1].astype(np.float64)