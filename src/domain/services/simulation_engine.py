import numpy as np
from ports.database_port import ClientRepository
from ports.ml_port import ProbabilityOfDefaultInferencePort
from domain.model.portfolio_containers import PortfolioSimulationChunk

class CreditPortfolioSimulationEngine:
    def __init__(self, db_repo: ClientRepository, inference_adapter: ProbabilityOfDefaultInferencePort, save_dir: str):
        self._db_repo = db_repo
        self._inference_adapter = inference_adapter
        self._save_dir = save_dir

    def run_portfolio_simulation(self, chunk_size: int = 50000) -> None:
        self._inference_adapter.load_model_asset(self._save_dir)
        
        print(f"Beginning portfolio credit simulation loop using a chunk partition size of: {chunk_size}")
        
        for chunk_df in self._db_repo.stream_simulation_chunks(chunk_size=chunk_size):
            pd_vector = self._inference_adapter.predict_probabilities_direct(chunk_df)
            
            upb_vector = chunk_df["current_actual_upb"].to_numpy(dtype=np.float64)
            sector_vector = chunk_df["sector_id"].to_numpy(dtype=np.int32)
            
            sim_chunk = PortfolioSimulationChunk(
                upb=upb_vector,
                sectors=sector_vector,
                pd_vector=pd_vector
            )
            
            self._execute_stochastic_math(sim_chunk)

    def _execute_stochastic_math(self, chunk: PortfolioSimulationChunk) -> None:
        pass