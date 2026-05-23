import pandas as pd
from sqlalchemy import create_engine, text, Engine
from pandas import DataFrame
import numpy as np

from config.configuration_objects import DBConfig, DataConfig
from ports.database_port import DataWriterPort

"""
This is etl pipeline for importing loan data from: Fannie Mae Single-Family Loan Performance Data, which is about 4 gb
"""

def generate_sample_beta_coeffs(df: DataFrame)->DataFrame:
    # todo: create a separate file/function for creating correlation coefficients for sectors
    return df.assign(asset_correlation=0.15, beta_economy=1.0)

def add_random_number(df: DataFrame) -> DataFrame:
    rng = np.random.default_rng()
    return df.assign(split=rng.random(len(df), dtype=np.float32))

def load_csv_to_DataFrame(data_config:DataConfig) -> tuple[DataFrame, DataFrame]:
    mapping = data_config.column_mapping
    cols_to_import = list(mapping.keys())
    print(f"Reading data from {data_config.csv_file_path}...")
    df = pd.read_csv(
        data_config.csv_file_path, 
        header=None, 
        usecols=cols_to_import,
        sep="|"
    ).rename(columns=mapping)
    # keep loans with highest delinquency status to determine default
    df["temp_date"] = pd.to_datetime(df["loan_report_date"].astype(str).str.zfill(6), format="%m%Y")
    df["numeric_delq"] = pd.to_numeric(df["current_loan_delinquency_status"], errors="coerce")
    df = df.sort_values(by=["loan_identifier", "numeric_delq", "temp_date"], na_position="first")
    loan_df = (df.drop_duplicates(subset=["loan_identifier"], keep="last")
               .drop(columns=["temp_date"]))

    #loan_df = (
        #df.sort_values(by=["loan_identifier", "temp_date"], ascending=[True, True])
        #.groupby("loan_identifier")
        #.tail(1)
        #.drop(columns=["temp_date"])
    #)

    loan_df["loan_age"] = loan_df["loan_age"].astype(str)
    # save date format MMYYYY as string
    loan_df["loan_report_date"] = loan_df["loan_report_date"].astype(str)

    unique_sectors = loan_df['property_type'].unique()
    # create second table for sectors
    sector_df = DataFrame({
        "property_type": unique_sectors,
        "sector_id": range(1, len(unique_sectors) + 1)
    })
    loan_df = loan_df.merge(sector_df, on="property_type", how="left").drop(columns=["property_type"])
    # add sector beta coefficients
    sector_df = generate_sample_beta_coeffs(sector_df)
    loan_df = add_random_number(loan_df)
    return (loan_df, sector_df)

def initialize_database(engine: Engine, schema_path: str) -> None:
    with open(schema_path, 'r') as file:
        query = file.read()
    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()
    print("Database schema initialized")

def load_csv(writer: DataWriterPort, data_config: DataConfig)->None:
    writer.initialize_schema(data_config.schema_file)
    loan_df, sector_df = load_csv_to_DataFrame(data_config)
    writer.write_portfolio_data(loan_df, sector_df)