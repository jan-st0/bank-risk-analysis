import os
import pandas as pd
from pandas import DataFrame
import numpy as np
from config.configuration_objects import DataConfig
from ports.database_port import DataWriterPort

def load_sector_configuration_matrix(config_dir: str) -> DataFrame:
    """
    Parses the static sector configuration matrix file containing 
    the systemic asset correlation and macro beta coefficients.
    """
    config_path = os.path.join(config_dir, "sectors_config.csv")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing mandatory framework configuration file: {config_path}")
        
    sector_df = pd.read_csv(
        config_path,
        sep="|",
        dtype={
            "property_type": str,
            "sector_id": np.int32,
            "beta_economy": np.float64,
            "asset_correlation": np.float64
        }
    )
    sector_df["property_type"] = sector_df["property_type"].str.strip().str.slice(0, 2)
    return sector_df

def _process_single_csv(csv_path: str, mapping: dict) -> DataFrame:
    cols_to_import = list(mapping.keys())
    
    df = pd.read_csv(
        csv_path, 
        header=None, 
        usecols=cols_to_import,
        sep="|"
    ).rename(columns=mapping)
    
    df["temp_date"] = pd.to_datetime(df["loan_report_date"].astype(str).str.zfill(6), format="%m%Y", errors="coerce")
    
    df["numeric_delq"] = pd.to_numeric(df["current_loan_delinquency_status"], errors="coerce").fillna(0).astype(np.int32)
    df["zb_code_clean"] = pd.to_numeric(df["zero_balance_code"], errors="coerce").fillna(0).astype(np.int32)
    
    df.loc[df["zb_code_clean"].isin([3, 6, 9, 15]), "numeric_delq"] = 99
    
    df_sorted = df.sort_values(by=["loan_identifier", "temp_date"], ascending=[True, True])
    
    base_aggregation_rules = {
        col: "first" for col in df_sorted.columns 
        if col not in ["loan_identifier", "numeric_delq", "temp_date", "current_loan_delinquency_status", "zb_code_clean"]
    }
    base_aggregation_rules["numeric_delq"] = "max"
    
    loan_df = df_sorted.groupby("loan_identifier", as_index=False).agg(base_aggregation_rules)
    loan_df = loan_df.rename(columns={"numeric_delq": "current_loan_delinquency_status"})
    
    loan_df["loan_identifier"] = loan_df["loan_identifier"].astype(np.int64)
    loan_df["current_actual_upb"] = pd.to_numeric(loan_df["current_actual_upb"], errors="coerce").fillna(0.0).astype(np.float64)
    loan_df["borrower_credit_score"] = pd.to_numeric(loan_df["borrower_credit_score"], errors="coerce").fillna(0).astype(np.int32)
    loan_df["ltv_ratio"] = pd.to_numeric(loan_df["ltv_ratio"], errors="coerce").fillna(0).astype(np.int8)
    loan_df["debt_to_income"] = pd.to_numeric(loan_df["debt_to_income"], errors="coerce").fillna(0).astype(np.int8)
    loan_df["current_loan_delinquency_status"] = loan_df["current_loan_delinquency_status"].astype(np.int8)
    loan_df["zero_balance_code"] = pd.to_numeric(loan_df["zero_balance_code"], errors="coerce").fillna(0).astype(np.int8)
    loan_df["loan_age"] = pd.to_numeric(loan_df["loan_age"], errors="coerce").fillna(0).astype(np.int8)
    loan_df["loan_report_date"] = pd.to_numeric(loan_df["loan_report_date"], errors="coerce").fillna(0).astype(np.int32)
    loan_df["occupancy_status"] = loan_df["occupancy_status"].astype(str).str.strip().str.slice(0, 1)
    loan_df["property_type"] = loan_df["property_type"].astype(str).str.strip().str.slice(0, 2)
    
    return loan_df

def load_csv_to_DataFrame(data_config: DataConfig, data_dir: str = "data") -> tuple[DataFrame, DataFrame, DataFrame]:
    mapping = data_config.column_mapping
    
    train_loan_df = _process_single_csv(data_config.training_csv_path, mapping)
    sim_loan_df = _process_single_csv(data_config.simulation_csv_path, mapping)
    
    sector_df = load_sector_configuration_matrix(data_dir)
    
    train_loan_df = train_loan_df.merge(sector_df[["property_type", "sector_id"]], on="property_type", how="left")
    sim_loan_df = sim_loan_df.merge(sector_df[["property_type", "sector_id"]], on="property_type", how="left")
    
    train_loan_df = train_loan_df.drop(columns=["property_type"])
    sim_loan_df = sim_loan_df.drop(columns=["property_type"])
    
    train_loan_df["sector_id"] = train_loan_df["sector_id"].fillna(0).astype(np.int32)
    sim_loan_df["sector_id"] = sim_loan_df["sector_id"].fillna(0).astype(np.int32)
    
    return (sim_loan_df, train_loan_df, sector_df)

def load_csv(writer: DataWriterPort, data_config: DataConfig) -> None:
    writer.initialize_schema(data_config.schema_file)
    sim_loan_df, train_loan_df, sector_df = load_csv_to_DataFrame(data_config)
    writer.write_portfolio_data(sim_loan_df, train_loan_df, sector_df)