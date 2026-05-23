from typing import Iterator, Protocol
from pandas import DataFrame
from domain.model.portfolio_containers import PortfolioClientRaw

class ClientRepository(Protocol):
    def stream_portfolio_chunks(self, chunks_count: int) -> Iterator[PortfolioClientRaw]:
        """
        Streams a unified record chunk container spanning continuous blocks.
        """
        ...

class DataWriterPort(Protocol):
    """
    Database port creating tables and writing data
    """
    def initialize_schema(self, schema_path: str) -> None:
        ...
    def write_portfolio_data(self, loan_df: DataFrame, sector_df: DataFrame) -> None:
        ...