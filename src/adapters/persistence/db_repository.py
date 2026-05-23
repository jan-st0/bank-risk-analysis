from typing import Iterator
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine, text
import numpy as np
from config.configuration_objects import DataConfig
from domain.model.portfolio_containers import PortfolioClientRaw, PORTFOLIO_STRUCTURED_DTYPE
from ports.database_port import ClientRepository, DataWriterPort


class SQLAlchemyClientRepository(ClientRepository, DataWriterPort):
    
    def __init__(self, conn_str: str, data_config: DataConfig):
        self._engine = create_engine(conn_str)
        self._loan_table_name = data_config.loan_table_name
        self._sector_table_name = data_config.sector_table_name
        self._config = data_config
        
        self._target_columns = list(data_config.column_mapping.values())
        if "sector_id" not in self._target_columns:
            self._target_columns.append("sector_id")
        if "split" not in self._target_columns:
            self._target_columns.append("split")

    def _compile_dynamic_query(self) -> str:
        columns_clause = ", ".join(self._target_columns)
        return f"""
            SELECT {columns_clause} 
            FROM {self._loan_table_name}
            WHERE split >= :low AND split < :high
        """

    def stream_portfolio_chunks(self, chunks_count: int) -> Iterator[PortfolioClientRaw]:
        split_intervals = np.linspace(0.0, 1.0, chunks_count + 1)
        raw_sql_query = self._compile_dynamic_query()

        with self._engine.connect() as conn:
            for i in range(chunks_count):
                low = float(split_intervals[i])
                high = float(split_intervals[i+1])
                
                df = pd.read_sql_query(
                    text(raw_sql_query), 
                    conn, 
                    params={"low": low, "high": high}
                )

                if df.empty:
                    continue

                chunk_size = len(df)
                structured_buffer = np.empty(chunk_size, dtype=PORTFOLIO_STRUCTURED_DTYPE)

                for field_name in PORTFOLIO_STRUCTURED_DTYPE.names:
                    target_dtype = PORTFOLIO_STRUCTURED_DTYPE[field_name]
                    
                    if field_name in df.columns:
                        structured_buffer[field_name] = df[field_name].to_numpy(
                            dtype=target_dtype, 
                            copy=False
                        )
                    else:
                        structured_buffer[field_name] = np.zeros(chunk_size, dtype=target_dtype)

                yield PortfolioClientRaw(structured_buffer)
    
    def initialize_schema(self, schema_path: str) -> None:
        with open(schema_path, "r") as schema_file:
            query = schema_file.read()
        with self._engine.connect() as conn:
            conn.execute(text(query))
            conn.commit()
    
    def write_portfolio_data(self, loan_df: DataFrame, sector_df: DataFrame) -> None:
        with self._engine.begin() as conn:
            loan_df.to_sql(self._loan_table_name, conn, if_exists="replace", index=False)
            sector_df.to_sql(self._sector_table_name, conn, if_exists="replace", index=False)