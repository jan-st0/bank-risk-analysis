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
class MLConfig:
    n_estimators: int
    learning_rate: float
    max_depth: int
    eval_metric: str
    n_jobs: int
    random_state: int
    calibration_method: str
    calibration_cv: int

@dataclass
class RootConfig:
    db_config: DBConfig
    data_config: DataConfig
    ml_config: MLConfig