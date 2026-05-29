from typing import Protocol, Generator
import pandas as pd

class ClientRepository(Protocol):
    def fetch_entire_training_set(self) -> pd.DataFrame:
        """Retrieves the full training dataset into application memory as a structured dataframe."""
        ...

    def stream_simulation_chunks(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Streams the simulation portfolio incrementally to optimize memory utilization."""
        ...

class DataWriterPort(Protocol):
    def initialize_schema(self, schema_path: str) -> None:
        """Executes targeted DDL scripts to reset cluster structures."""
        ...

    def write_portfolio_data(
        self, 
        sim_loan_df: pd.DataFrame, 
        train_loan_df: pd.DataFrame, 
        sector_df: pd.DataFrame
    ) -> None:
        ...