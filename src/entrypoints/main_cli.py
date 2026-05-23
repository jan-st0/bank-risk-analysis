import argparse
from adapters.persistence.db_repository import SQLAlchemyClientRepository
from adapters.persistence.load_csv import load_csv
from config.configuration_objects import RootConfig
from config.load_root_config import load_config
def cli_run():
    parser = argparse.ArgumentParser(description="SCRMVF v2.0 Runner")
    parser.add_argument(
        "task",
        choices=["etl", "simulate", "report"],
        help="Task to perform",
    )
    args = parser.parse_args()

    cfg: RootConfig  = load_config()

    if args.task == "etl":
        db_adapter = SQLAlchemyClientRepository(
            conn_str=cfg.db_config.conn_string, 
            data_config=cfg.data_config
        )
        load_csv(writer=db_adapter, data_config=cfg.data_config)
