import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import psutil

from ports.database_port import ClientRepository
from ports.ml_port import ProbabilityOfDefaultInferencePort
from domain.model.portfolio_containers import PortfolioSimulationChunk
from domain.model.math_engine import IndependentMonteCarloEngine
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

    @staticmethod
    def calculate_dynamic_chunk_size(num_simulations: int, num_workers: int, memory_utilization_target: float = 0.8) -> int:
        available_memory_bytes = psutil.virtual_memory().available
        allocated_memory_budget = available_memory_bytes * memory_utilization_target
        
        base_process_overhead = num_workers * 350 * 1024 * 1024
        usable_simulation_budget = max(0, allocated_memory_budget - base_process_overhead)
        
        bytes_per_row_simulated = num_simulations * 8
        real_world_bytes_per_row = bytes_per_row_simulated * 4 
        
        total_affordable_rows = int(usable_simulation_budget / real_world_bytes_per_row)
        dynamic_chunk = int(total_affordable_rows / num_workers)
        
        return max(1000, min(dynamic_chunk, 10000))

    def run_portfolio_simulation(self, num_simulations: int, num_workers: int) -> None:
        self._inference_adapter.load_model_asset(self._save_dir)
        chunk_size = self.calculate_dynamic_chunk_size(num_simulations, num_workers)
        print(f"[SIMULATION START] Processing {num_simulations} paths across {num_workers} parallel forks with {chunk_size} chunk size...")
        
        global_loss_distribution = np.zeros(num_simulations, dtype=np.float64)
        t0 = time.time()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            asynchronous_futures = []
            
            for chunk_df in self._db_repo.stream_simulation_chunks(chunk_size=chunk_size):
                pd_vector = self._inference_adapter.predict_probabilities_direct(chunk_df)
                
                sim_chunk = PortfolioSimulationChunk(
                    upb=chunk_df["current_actual_upb"].to_numpy(dtype=np.float64),
                    pd_vector=pd_vector,
                    ltv=chunk_df["ltv_ratio"].to_numpy(dtype=np.int32)
                )
                
                future = executor.submit(
                    IndependentMonteCarloEngine.execute_chunk_simulation,
                    sim_chunk, num_simulations
                )
                asynchronous_futures.append(future)
            
            for completed_future in asynchronous_futures:
                global_loss_distribution += completed_future.result()

        print(f"[SIMULATION COMPLETE] Multi-process processing finished in {time.time() - t0:.2f}s.")
        self._presenter_adapter.render_loss_distribution_histogram(global_loss_distribution, self._save_dir)