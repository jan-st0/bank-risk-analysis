import os
import pandas as pd
from sqlalchemy import create_engine, text
from config.configuration_objects import DataConfig
from ports.database_port import ClientRepository, DataWriterPort

class PostgreSQLUnifiedRepository(ClientRepository, DataWriterPort):
    def __init__(self, conn_str: str, data_config: DataConfig):
        self._engine = create_engine(conn_str)
        self._config = data_config

    def _build_extraction_query(self, source_table: str) -> str:
        return f"""
            SELECT 
                l.loan_identifier, l.current_actual_upb, l.loan_report_date,
                l.current_loan_delinquency_status, l.zero_balance_code,
                l.borrower_credit_score, l.ltv_ratio, l.debt_to_income,
                l.occupancy_status, l.loan_age, l.sector_id, s.property_type,
                s.beta_economy, s.asset_correlation
            FROM {source_table} l
            LEFT JOIN {self._config.sector_table_name} s ON l.sector_id = s.sector_id
        """

    def fetch_entire_training_set(self) -> pd.DataFrame:
        query = self._build_extraction_query(self._config.training_table_name)
        with self._engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
        return df

    def stream_simulation_chunks(self, chunk_size: int):
        query = self._build_extraction_query(self._config.simulation_table_name)
        with self._engine.connect() as conn:
            streaming_conn = conn.execution_options(stream_results=True)
            proxy = streaming_conn.execute(text(query))
            while True:
                rows = proxy.fetchmany(chunk_size)
                if not rows:
                    break
                yield pd.DataFrame(rows, columns=proxy.keys())

    def initialize_schema(self, schema_path: str) -> None:
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Target SQL schema file definition missing at: {schema_path}")
        with open(schema_path, "r") as schema_file:
            query = schema_file.read()
            
        with self._engine.connect() as conn:
            conn.execute(text(query))
            conn.commit()

    def write_portfolio_data(
        self, 
        sim_loan_df: pd.DataFrame, 
        train_loan_df: pd.DataFrame, 
        sector_df: pd.DataFrame
    ) -> None:

        with self._engine.begin() as conn:
            truncate_query = f"""
                TRUNCATE TABLE 
                    {self._config.simulation_table_name}, 
                    {self._config.training_table_name}, 
                    {self._config.sector_table_name} 
                CASCADE;
            """
            conn.execute(text(truncate_query))
            
            sector_df.to_sql(self._config.sector_table_name, conn, if_exists="append", index=False)
            sim_loan_df.to_sql(self._config.simulation_table_name, conn, if_exists="append", index=False)
            train_loan_df.to_sql(self._config.training_table_name, conn, if_exists="append", index=False)