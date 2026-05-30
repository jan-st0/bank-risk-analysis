import numpy as np
import pytest
from domain.model.portfolio_containers import PortfolioSimulationChunk
from domain.model.math_engine import VasicekMonteCarloEngine, StochasticLGDModel


@pytest.fixture
def sample_simulation_chunk() -> PortfolioSimulationChunk:
    """
    Constructs a deterministic, micro-scale PortfolioSimulationChunk payload.
    Simulates 3 distinct credit profiles mapping across different sectors.
    """
    upb = np.array([500_000.0, 250_000.0, 1_200_000.0], dtype=np.float64)
    sectors = np.array([0, 1, 0], dtype=np.int32)  # Sector indices matching rows
    pd_vector = np.array([0.02, 0.05, 0.01], dtype=np.float64)  # Base PD thresholds
    beta = np.array([0.15, 0.22, 0.15], dtype=np.float64)  # Macroeconomic beta sensitivities
    rho = np.array([0.10, 0.15, 0.10], dtype=np.float64)  # sector asset correlations
    ltv = np.array([75, 90, 60], dtype=np.int32)  # Loan-to-value markers for LGD scaling
    
    return PortfolioSimulationChunk(
        upb=upb,
        sectors=sectors,
        pd_vector=pd_vector,
        beta=beta,
        rho=rho,
        ltv=ltv
    )


def test_stochastic_lgd_model_bounds():
    """
    Validates that the Beta distribution method of moments calculation within
    StochasticLGDModel handles vector arithmetic safely and clips outputs inside [0.10, 0.99].
    """
    model = StochasticLGDModel(baseline_lgd=0.45, variance_factor=0.05)
    prng = np.random.Generator(np.random.PCG64(12345))
    
    # Test extreme input conditions to verify maximum/minimum clipping paths
    ltv_vector = np.array([0, 80, 500], dtype=np.int32)
    lgd_vector = model.calculate_lgd_vector(ltv_vector, prng)
    
    assert lgd_vector.shape == (3,)
    assert lgd_vector.dtype == np.float64
    # Ensure all elements fall strictly within the theoretical domain definitions
    assert np.all(lgd_vector >= 0.10)
    assert np.all(lgd_vector <= 0.99)


def test_vasicek_monte_carlo_engine_execution(sample_simulation_chunk):
    """
    Performs an integration test over the vectorized multi-factor portfolio simulation engine.
    Verifies state tracking, execution consistency, and standard output dimensionality.
    """
    num_simulations = 1000
    num_sectors = 2
    
    # Instantiate global systemic shocks using a fixed seed state
    global_prng = np.random.Generator(np.random.PCG64(42))
    y_global = global_prng.standard_normal(num_simulations, dtype=np.float64)
    z_global = global_prng.standard_normal((num_sectors, num_simulations), dtype=np.float64)
    
    # Isolate child seeds to insulate the underlying parallel workers from dependency overlap
    master_seed_seq = np.random.SeedSequence(99999)
    
    # Execute the integration pipeline over the isolated sample chunk
    portfolio_loss_distribution = VasicekMonteCarloEngine.execute_chunk_simulation(
        chunk=sample_simulation_chunk,
        y_global=y_global,
        z_global=z_global,
        seed_sequence=master_seed_seq
    )
    
    # Verify shape consistency matching simulation paths
    assert portfolio_loss_distribution.shape == (num_simulations,)
    assert portfolio_loss_distribution.dtype == np.float64
    
    # Portfolio losses cannot be negative or exceed the total aggregate Unpaid Principal Balance (UPB)
    total_portfolio_upb = np.sum(sample_simulation_chunk.unpaid_principal_balance)
    assert np.all(portfolio_loss_distribution >= 0.0)
    assert np.all(portfolio_loss_distribution <= total_portfolio_upb)
    
    # Run a secondary pass with an identical seed state to confirm mathematical invariance
    identical_loss_distribution = VasicekMonteCarloEngine.execute_chunk_simulation(
        chunk=sample_simulation_chunk,
        y_global=y_global,
        z_global=z_global,
        seed_sequence=master_seed_seq
    )
    np.testing.assert_array_equal(portfolio_loss_distribution, identical_loss_distribution)