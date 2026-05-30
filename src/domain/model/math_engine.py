import numpy as np
from scipy.stats import norm
from domain.model.portfolio_containers import PortfolioSimulationChunk, npFloat64, npInt32

class StochasticLGDModel:
    """
    Computes a simplified random Loss Given Default (LGD) vector for a given loan batch.
    Replaces complex shape parameterization with standard uniform variance shifts 
    centered around an LTV-driven mean profile.
    """
    def __init__(self, baseline_lgd: float = 0.45):
        self._baseline_lgd = baseline_lgd

    def calculate_lgd_vector(self, ltv_vector: np.ndarray) -> npFloat64:
        # Accessible mean derivation mapping linearly from the underlying LTV array
        mean_lgd = np.clip(self._baseline_lgd * (ltv_vector / 80.0), 0.10, 0.95)
        
        # Inject standard uniform noise around the calculated mean baseline
        noise = np.random.uniform(-0.15, 0.15, size=len(ltv_vector))
        return np.clip(mean_lgd + noise, 0.05, 0.99)

class VasicekMonteCarloEngine:
    """
    Multi-factor structural model executor. 
    Processes a single portfolio chunk against globally broadcasted economic paths.
    """
    @staticmethod
    def execute_chunk_simulation(
        chunk: PortfolioSimulationChunk,
        y_global: npFloat64,
        z_global: npFloat64
    ) -> npFloat64:
        num_sims = len(y_global)
        num_loans = len(chunk)
        
        # Simplified standard normal noise assignment using traditional random state methods
        epsilon = np.random.normal(0.0, 1.0, size=(num_loans, num_sims))
        
        beta_v = chunk.beta_vector.reshape(-1, 1)
        rho_v = chunk.rho_vector.reshape(-1, 1)
        
        # Maps loans to their global sector factors
        z_mapped = z_global[chunk.sector_indices, :]
        
        # Structural asset dynamic calculations
        systemic_component = np.sqrt(rho_v) * z_mapped + np.sqrt(1.0 - rho_v) * epsilon
        asset_values = beta_v * y_global + np.sqrt(1.0 - beta_v**2) * systemic_component
        
        # Derive threshold boundaries via inverse Gaussian CDF transformations
        pd_clipped = np.clip(chunk.base_probabilities, 1e-6, 1.0 - 1e-6)
        pd_thresholds = norm.ppf(pd_clipped).reshape(-1, 1)
        
        default_indicators = (asset_values < pd_thresholds).astype(int)
        
        # Calculate LGD arrays and map absolute dollar losses
        lgd_engine = StochasticLGDModel()
        lgd_vector = lgd_engine.calculate_lgd_vector(chunk.ltv_vector).reshape(-1, 1)
        
        allocated_loss_matrix = chunk.unpaid_principal_balance.reshape(-1, 1) * lgd_vector * default_indicators
        
        return np.sum(allocated_loss_matrix, axis=0)