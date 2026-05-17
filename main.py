import argparse
from hydra import compose, initialize
from src.infrastructure.load import load_csv

def main():
    parser = argparse.ArgumentParser(description="SCRMVF v2.0 Runner")
    parser.add_argument('task', choices=['etl', 'simulate', 'report', 'csv-imp'], 
                        help="Task to perform")
    args = parser.parse_args()

    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="config")

        if args.task == 'etl':
            load_csv(cfg)

if __name__ == "__main__":
    main()
