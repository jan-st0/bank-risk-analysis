import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from config.configuration_objects import DataConfig
from adapters.persistence.load_csv import _process_single_csv, load_csv_to_DataFrame


@pytest.fixture
def mock_etl_environment():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write static sectors
        sectors_path = os.path.join(tmpdir, "sectors_config.csv")
        with open(sectors_path, "w") as f:
            f.write("property_type|sector_id|beta_economy|asset_correlation\n")
            f.write("SF|1|0.12|0.08\n")
            f.write("CO|2|0.15|0.10\n")

        # These numbers correspond directly to 0-indexed column positions when read by pandas
        column_mapping = {
            10: "current_actual_upb", 
            26: "property_type", 
            38: "current_loan_delinquency_status",
            42: "zero_balance_code", 
            22: "borrower_credit_score", 
            18: "ltv_ratio",
            21: "debt_to_income", 
            28: "occupancy_status", 
            14: "loan_age",
            0: "loan_identifier", 
            1: "loan_report_date"
        }

        matrix = np.empty((2, max(column_mapping.keys())+1), dtype=object)
        matrix[:] = ""
        
        # Row 0: Normal snapshot (idx 0-based matching mapping keys exactly)
        matrix[0, 0] = 10001      # loan_identifier
        matrix[0, 1] = "072025"   # loan_report_date
        matrix[0, 10] = 180000.0  # current_actual_upb
        matrix[0, 14] = 12        # loan_age
        matrix[0, 18] = 80        # ltv_ratio
        matrix[0, 21] = 45        # debt_to_income
        matrix[0, 22] = 720       # borrower_credit_score
        matrix[0, 26] = "SF"      # property_type
        matrix[0, 28] = "O"       # occupancy_status
        matrix[0, 38] = 0         # current_loan_delinquency_status
        matrix[0, 42] = 0         # zero_balance_code

        # Row 1: Default snapshot for same loan identifier
        matrix[1, 0] = 10001
        matrix[1, 1] = "082025"
        matrix[1, 10] = 175000.0
        matrix[1, 14] = 13
        matrix[1, 18] = 81
        matrix[1, 21] = 45
        matrix[1, 22] = 715
        matrix[1, 26] = "SF"
        matrix[1, 28] = "O"
        matrix[1, 38] = 2
        matrix[1, 42] = 3         # Remaps down to structural default code 99

        train_csv = os.path.join(tmpdir, "train_mock.csv")
        pd.DataFrame(matrix).to_csv(train_csv, sep="|", header=False, index=False)

        # Simulation vector block (Single row)
        sim_matrix = np.empty((1, 50), dtype=object)
        sim_matrix[:] = ""
        sim_matrix[0, 0] = 20002
        sim_matrix[0, 1] = "102025"
        sim_matrix[0, 10] = 320000.0
        sim_matrix[0, 14] = 5
        sim_matrix[0, 18] = 75
        sim_matrix[0, 21] = 30
        sim_matrix[0, 22] = 790
        sim_matrix[0, 26] = "CO"
        sim_matrix[0, 28] = "I"
        sim_matrix[0, 38] = 0
        sim_matrix[0, 42] = 0

        sim_csv = os.path.join(tmpdir, "sim_mock.csv")
        pd.DataFrame(sim_matrix).to_csv(sim_csv, sep="|", header=False, index=False)

        data_config = DataConfig(
            training_csv_path=train_csv, 
            simulation_csv_path=sim_csv,
            column_mapping=column_mapping, 
            simulation_table_name="sim_table",
            training_table_name="train_table", 
            sector_table_name="sectors",
            schema_file="data/db_schema.sql", 
            model_save_dir="models/"
        )
        yield data_config, tmpdir


def test_process_single_csv_aggregation_and_remapping(mock_etl_environment):
    data_config, _ = mock_etl_environment
    processed_df = _process_single_csv(data_config.training_csv_path, data_config.column_mapping)
    assert len(processed_df) == 1
    assert processed_df.loc[0, "loan_identifier"] == 10001
    assert processed_df.loc[0, "current_loan_delinquency_status"] == 99


def test_load_csv_to_dataframe_relational_mapping(mock_etl_environment):
    data_config, data_dir = mock_etl_environment
    sim_df, train_df, sector_df = load_csv_to_DataFrame(data_config, data_dir=data_dir)
    assert "sector_id" in train_df.columns
    assert train_df.loc[0, "sector_id"] == 1
    assert sim_df.loc[0, "sector_id"] == 2