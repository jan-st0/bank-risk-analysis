from dataclasses import dataclass


@dataclass
class DBConfig:
    conn_string: str

@dataclass
class DataConfig:
    csv_file_path: str
    column_mapping: dict[int, str]
    loan_table_name: str
    sector_table_name: str
    schema_file: str

@dataclass
class RootConfig:
    db_config: DBConfig
    data_config: DataConfig