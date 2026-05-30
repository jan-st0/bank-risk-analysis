import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from ports.database_port import ClientRepository
from ports.ml_port import ProbabilityOfDefaultInferencePort
from domain.model.portfolio_containers import PortfolioSimulationChunk
from domain.model.math_engine import VasicekMonteCarloEngine
from ports.presentation_port import RenderVisualPresentation

class CreditPortfolioSimulationEngine:
    def __init__(self, db_repo: ClientRepository, inference_adapter: ProbabilityOfDefaultInferencePort, save_dir: str, present_adapter: RenderVisualPresentation):
        self._db_repo = db_repo
        self._inference_adapter = inference_adapter
        self._save_dir = save_dir
        self._presenter_adapter = present_adapter

    def ensure_model_exists(self, training_orchestrator) -> None:
        model_path = os.path.join(self._save_dir, "calibrated_pd_xgboost.pkl")
        if not os.path.exists(model_path):
            print(f"[SYSTEM REGISTRATION] Model asset '{model_path}' not found on disk.")
            print("[SYSTEM REGISTRATION] Executing baseline training pipeline orchestrator dynamically...")
            training_orchestrator.execute_training_pipeline()

    def run_portfolio_simulation(self, chunk_size: int, num_simulations: int, num_workers: int) -> None:
        self._inference_adapter.load_model_asset(self._save_dir)
        print(f"[SIMULATION START] Processing {num_simulations} paths across {num_workers} parallel forks...")
        
        # Establish global systematic factors
        y_global = np.random.normal(0.0, 1.0, size=num_simulations)
        z_global = np.random.normal(0.0, 1.0, size=(100, num_simulations))
        
        global_loss_distribution = np.zeros(num_simulations, dtype=np.float64)
        t0 = time.time()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            asynchronous_futures = []
            
            for chunk_df in self._db_repo.stream_simulation_chunks(chunk_size=chunk_size):
                pd_vector = self._inference_adapter.predict_probabilities_direct(chunk_df)
                
                sim_chunk = PortfolioSimulationChunk(
                    upb=chunk_df["current_actual_upb"].to_numpy(dtype=np.float64),
                    sectors=chunk_df["sector_id"].to_numpy(dtype=np.int32),
                    pd_vector=pd_vector,
                    beta=chunk_df["beta_economy"].to_numpy(dtype=np.float64),
                    rho=chunk_df["asset_correlation"].to_numpy(dtype=np.float64),
                    ltv=chunk_df["ltv_ratio"].to_numpy(dtype=np.int32)
                )
                
                future = executor.submit(
                    VasicekMonteCarloEngine.execute_chunk_simulation,
                    sim_chunk, y_global, z_global
                )
                asynchronous_futures.append(future)
            
            for completed_future in asynchronous_futures:
                global_loss_distribution += completed_future.result()

        print(f"[SIMULATION COMPLETE] Multi-process processing finished in {time.time() - t0:.2f}s.")
        self._presenter_adapter.render_loss_distribution_histogram(global_loss_distribution, self._save_dir)