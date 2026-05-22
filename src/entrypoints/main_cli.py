import argparse
from adapters.load_csv import load_csv
from config.configuration_objects import RootConfig
from config import load_root_config

def cli_run():
    parser = argparse.ArgumentParser(description="SCRMVF v2.0 Runner")
    parser.add_argument(
        "task",
        choices=["etl", "simulate", "report"],
        help="Task to perform",
    )
    args = parser.parse_args()

    cfg: RootConfig  = load_root_config()

    if args.task == "etl":
        load_csv(cfg.secrets.db, cfg.app.data)
