import argparse
from adapters.persistence.db_repository import PostgreSQLUnifiedRepository
from adapters.persistence.load_csv import load_csv
from adapters.ml.pd_model_adapter import VectorizedCalibratedXGBoostAdapter
from domain.services.model_training_engine import ModelTrainingOrchestrator
from domain.services.simulation_engine import CreditPortfolioSimulationEngine
from config.configuration_objects import RootConfig
from config.load_root_config import load_config

def cli_run():
    parser = argparse.ArgumentParser(description="SCRMVF v2.0 Production Engine")
    parser.add_argument(
        "task",
        choices=["etl", "train", "simulate"],
        help="Target execution task routine to run",
    )
    args = parser.parse_args()
    cfg: RootConfig = load_config()

    db_adapter = PostgreSQLUnifiedRepository(
        conn_str=cfg.db_config.conn_string, 
        data_config=cfg.data_config
    )

    ml_adapter = VectorizedCalibratedXGBoostAdapter()

    if args.task == "etl":
        print("[ETL MODULE] Commencing raw portfolio data processing and data ingestion...")
        load_csv(writer=db_adapter, data_config=cfg.data_config)
        print("[ETL MODULE] Systemic staging and data insertion completed successfully.")

    elif args.task == "train":
        orchestrator = ModelTrainingOrchestrator(
            db_repository=db_adapter,
            ml_trainer=ml_adapter,
            save_dir=cfg.data_config.model_save_dir
        )
        orchestrator.execute_training_pipeline()
        
    elif args.task == "simulate":
        engine = CreditPortfolioSimulationEngine(
            db_repo=db_adapter,
            inference_adapter=ml_adapter,
            save_dir=cfg.data_config.model_save_dir
        )
        engine.run_portfolio_simulation(chunk_size=100000)