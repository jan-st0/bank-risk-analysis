import numpy as np
import pandas as pd
from ports.database_port import ClientRepository
from ports.ml_port import ModelTrainingPort
from domain.model.portfolio_containers import TrainingBatch
from config.configuration_objects import MLConfig

class ModelTrainingOrchestrator:
    def __init__(self, db_repository: ClientRepository, ml_trainer: ModelTrainingPort, save_dir: str, ml_config: MLConfig):
        self._db_repository = db_repository
        self._ml_trainer = ml_trainer
        self._save_dir = save_dir
        self._ml_config = ml_config

    def _derive_targets_and_filter_leakage(self, df: pd.DataFrame) -> TrainingBatch:
        if df.empty:
            raise ValueError("The query returned an empty dataset; cannot execute training.")
        
        targets = np.where(
            (df["current_loan_delinquency_status"] >= 3) | 
            (df["current_loan_delinquency_status"] == 99), 
            1, 0
        ).astype(np.int32)

        leakage_cols = [
            "loan_identifier", "loan_report_date", "current_loan_delinquency_status", 
            "zero_balance_code", "sector_id", "numeric_delq", "zb_code_clean", "temp_date"
        ]
        feature_df = df.drop(columns=leakage_cols, errors="ignore")
        
        return TrainingBatch(
            features=feature_df,
            targets=targets,
            feature_names=feature_df.columns.tolist()
        )

    def execute_training_pipeline(self) -> None:
        raw_df = self._db_repository.fetch_entire_training_set()
        domain_batch = self._derive_targets_and_filter_leakage(raw_df)
        
        metrics = self._ml_trainer.train_and_calibrate(
            batch=domain_batch, 
            save_dir=self._save_dir, 
            ml_config=self._ml_config
        )
        
        for metric, val in metrics.items():
            print(f"[METRIC VALIDATION] {metric.upper()}: {val:.6f}")