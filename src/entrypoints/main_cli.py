import argparse
from hydra import compose, initialize
from hydra.utils import instantiate

from src.adapters.load import load_csv
from src.config import RootConfig


def main():
    parser = argparse.ArgumentParser(description="SCRMVF v2.0 Runner")
    parser.add_argument(
        "task",
        choices=["etl", "simulate", "report"],
        help="Task to perform",
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../../config"):
        hydra_cfg = compose(config_name="config")

        cfg: RootConfig  = instantiate(hydra_cfg)

    if args.task == "etl":
        load_csv(cfg.secrets.db, cfg.app.data)


if __name__ == "__main__":
    main()