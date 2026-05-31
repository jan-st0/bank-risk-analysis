import numpy as np
import pytest
from domain.model.portfolio_containers import PortfolioSimulationChunk
from domain.model.math_engine import IndependentMonteCarloEngine, DeterministicLGDModel

@pytest.fixture
def sample_simulation_chunk() -> PortfolioSimulationChunk:
    upb = np.array([500_000.0, 250_000.0, 1_200_000.0], dtype=np.float64)
    pd_vector = np.array([0.02, 0.05, 0.01], dtype=np.float64)
    ltv = np.array([75, 95, 50], dtype=np.int32)
    
    return PortfolioSimulationChunk(
        upb=upb,
        pd_vector=pd_vector,
        ltv=ltv
    )

def test_deterministic_lgd_model_bounds():
    """Validates the exact piecewise threshold logic for LGD."""
    ltv_vector = np.array([50, 75, 85, 95], dtype=np.int32)
    lgd_vector = DeterministicLGDModel.calculate_lgd_vector(ltv_vector)
    
    assert lgd_vector.shape == (4,)
    np.testing.assert_array_almost_equal(lgd_vector, [0.20, 0.35, 0.50, 0.70])

def test_independent_monte_carlo_engine_execution(sample_simulation_chunk):
    """
    Executes the vectorized Bernoulli default framework. 
    Validates matrix dimensionality and strict bounds on allocated loss.
    """
    num_simulations = 1000
    
    np.random.seed(999)
    portfolio_loss_distribution = IndependentMonteCarloEngine.execute_chunk_simulation(
        chunk=sample_simulation_chunk,
        num_sims=num_simulations
    )
    
    assert portfolio_loss_distribution.shape == (num_simulations,)
    assert portfolio_loss_distribution.dtype == np.float64
    
    # Total portfolio losses bounded [0, aggregate UPB]
    total_portfolio_upb = np.sum(sample_simulation_chunk.unpaid_principal_balance)
    assert np.all(portfolio_loss_distribution >= 0.0)
    assert np.all(portfolio_loss_distribution <= total_portfolio_upb)