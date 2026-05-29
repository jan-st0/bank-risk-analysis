import numpy as np
from scipy.stats import norm
from domain.model.portfolio_containers import PortfolioSimulationChunk, npFloat64, npInt32

class StochasticLGDModel:
    def __init__(self, baseline_lgd: float = 0.45, variance_factor: float = 0.05):
        self._baseline_lgd = baseline_lgd
        self._variance = variance_factor

    def calculate_lgd_vector(self, ltv_vector: npInt32, generator: np.random.Generator) -> npFloat64:
        mean_lgd = np.clip(self._baseline_lgd * (ltv_vector / 80.0), 0.10, 0.99)

        alpha = mean_lgd * ((mean_lgd * (1 - mean_lgd) / self._variance) - 1)

        beta_param = (1 - mean_lgd) * ((mean_lgd * (1 - mean_lgd) / self._variance) - 1)

        alpha = np.maximum(alpha, 1.0)

        beta_param = np.maximum(beta_param, 1.0)

        return generator.beta(alpha, beta_param).astype(np.float64)

class VasicekMonteCarloEngine:
    @staticmethod
    def execute_chunk_simulation(
        chunk: PortfolioSimulationChunk,
        y_global: npFloat64,
        z_global: npFloat64,
        seed_sequence: np.random.SeedSequence
    ) -> npFloat64:
        num_sims = y_global.shape[0]
        num_loans = len(chunk)
        
        prng = np.random.Generator(np.random.PCG64(seed_sequence))
        epsilon = prng.standard_normal((num_loans, num_sims), dtype=np.float64)
        
        beta_v = chunk.beta_vector[:, None]
        rho_v = chunk.rho_vector[:, None]
        
        z_mapped = z_global[chunk.sector_indices, :]
        
        systemic_sector_risk = np.sqrt(rho_v) * z_mapped
        idiosyncratic_risk = np.sqrt(1 - rho_v) * epsilon
        
        asset_values = beta_v * y_global + np.sqrt(1 - beta_v**2) * (systemic_sector_risk + idiosyncratic_risk)
        
        pd_thresholds = norm.ppf(np.clip(chunk.base_probabilities, 1e-9, 1 - 1e-9))[:, None]
        default_indicators = asset_values < pd_thresholds
        
        lgd_model = StochasticLGDModel()
        lgd_vector = lgd_model.calculate_lgd_vector(chunk.ltv_vector, prng)[:, None]
        
        loss_matrix = chunk.unpaid_principal_balance[:, None] * lgd_vector * default_indicators
        return np.sum(loss_matrix, axis=0)