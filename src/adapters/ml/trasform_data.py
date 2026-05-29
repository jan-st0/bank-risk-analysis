from domain.model.portfolio_containers import PortfolioClientRaw
from pandas import DataFrame
import pandas as pd

def transform_raw_client_data(raw_chunk: PortfolioClientRaw) -> DataFrame:
    df = pd.DataFrame(raw_chunk.data)
    df = df.drop(columns=["sector_id", "loan_identifier", "split"])
    df = 