import numpy as np
from domain.model.portfolio_containers import PortfolioSimulationChunk, npFloat64

class DeterministicLGDModel:
    """
    Computes Loss Given Default (LGD) vector based on LTV thresholds.
    """
    @staticmethod
    def calculate_lgd_vector(ltv_vector: np.ndarray) -> npFloat64:
        lgd = np.full(ltv_vector.shape, 0.35, dtype=np.float64)
        lgd[ltv_vector <= 60] = 0.20
        lgd[(ltv_vector > 80) & (ltv_vector <= 90)] = 0.50
        lgd[ltv_vector > 90] = 0.70
        return lgd

class IndependentMonteCarloEngine:
    """
    Executes independent Bernoulli trials for default simulation.
    """
    @staticmethod
    def execute_chunk_simulation(
        chunk: PortfolioSimulationChunk,
        num_sims: int
    ) -> npFloat64:
        num_loans = len(chunk)
        
        # Generate random uniform draws for independent Bernoulli trials
        random_draws = np.random.uniform(0.0, 1.0, size=(num_loans, num_sims))
        
        pd_vector = chunk.base_probabilities.reshape(-1, 1)
        
        # A loan defaults if the random draw is less than its predicted PD
        default_indicators = (random_draws < pd_vector).astype(np.int32)
        
        lgd_vector = DeterministicLGDModel.calculate_lgd_vector(chunk.ltv_vector).reshape(-1, 1)
        
        # Calculate absolute dollar losses
        allocated_loss_matrix = chunk.unpaid_principal_balance.reshape(-1, 1) * lgd_vector * default_indicators
        
        return np.sum(allocated_loss_matrix, axis=0)