import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from config.configuration_objects import MLConfig
from domain.model.portfolio_containers import TrainingBatch
from adapters.ml.pd_model_adapter import VectorizedCalibratedXGBoostAdapter


@pytest.fixture
def sample_ml_config() -> MLConfig:
    return MLConfig(
        n_estimators=2, learning_rate=0.1, max_depth=1, eval_metric="logloss",
        n_jobs=-1, random_state=42, calibration_method="isotonic", calibration_cv=2
    )


@pytest.fixture
def structured_dataframe() -> pd.DataFrame:
    """Provides a data scale capable of supporting stratified validation splitting."""
    return pd.DataFrame({
        "loan_identifier": list(range(101, 111)),
        "current_actual_upb": [250000.0, 110000.0, 300000.0, 0.0, 150000.0] * 2,
        "borrower_credit_score": [720, 680, 590, 810, 640] * 2,
        "ltv_ratio": [80, 95, 70, 60, 85] * 2,
        "debt_to_income": [35, 45, 50, 22, 38] * 2,
        "loan_age": [12, 24, 6, 48, 18] * 2,
        "property_type": ["SF", "CO", "SF", "CP", "MH"] * 2,
        "occupancy_status": ["O", "I", "O", "O", "I"] * 2,
        "current_loan_delinquency_status": [0, 0, 3, 0, 99] * 2,
        "zb_code_clean": [0, 0, 0, 0, 15] * 2
    })


def test_adapter_dataframe_vectorized_cleansing(structured_dataframe):
    adapter = VectorizedCalibratedXGBoostAdapter()
    cleaned_df = adapter.transform_raw_dataframe(structured_dataframe)
    assert cleaned_df["property_type"].iloc[0] == "SF"
    assert cleaned_df["current_actual_upb"].dtype == np.float64


def test_train_calibrate_and_direct_inference_loop(structured_dataframe, sample_ml_config):
    adapter = VectorizedCalibratedXGBoostAdapter()
    cleaned_df = adapter.transform_raw_dataframe(structured_dataframe)
    
    targets = np.where(
        (cleaned_df["current_loan_delinquency_status"] >= 3) | 
        (cleaned_df["current_loan_delinquency_status"] == 99), 1, 0
    ).astype(np.int32)
    
    feature_cols = [
        "current_actual_upb", "borrower_credit_score", "ltv_ratio", 
        "debt_to_income", "loan_age", "property_type", "occupancy_status"
    ]
    
    batch = TrainingBatch(features=cleaned_df[feature_cols], targets=targets, feature_names=feature_cols)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = adapter.train_and_calibrate(batch=batch, save_dir=tmpdir, ml_config=sample_ml_config)
        assert "roc_auc" in metrics
        assert os.path.exists(os.path.join(tmpdir, "calibrated_pd_xgboost.pkl"))
        
        inference_adapter = VectorizedCalibratedXGBoostAdapter()
        inference_adapter.load_model_asset(tmpdir)
        probabilities = inference_adapter.predict_probabilities_direct(structured_dataframe)
        
        assert probabilities.shape == (10,)
        assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)