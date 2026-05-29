from dataclasses import dataclass

@dataclass
class DBConfig:
    conn_string: str

@dataclass
class DataConfig:
    training_csv_path: str
    simulation_csv_path: str
    column_mapping: dict[int, str]
    simulation_table_name: str
    training_table_name: str
    sector_table_name: str
    schema_file: str
    model_save_dir: str

@dataclass
class RootConfig:
    db_config: DBConfig
    data_config: DataConfig