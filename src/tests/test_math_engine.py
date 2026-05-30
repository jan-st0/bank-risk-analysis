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
    sectors = np.array([0, 1, 0], dtype=np.int32)
    pd_vector = np.array([0.02, 0.05, 0.01], dtype=np.float64)
    beta = np.array([0.15, 0.22, 0.15], dtype=np.float64)
    rho = np.array([0.10, 0.15, 0.10], dtype=np.float64)
    ltv = np.array([75, 90, 60], dtype=np.int32)
    
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
    Validates the stochastic LGD mapping utilizing affine LTV transformations 
    and uniform noise injections. Enforces the systemic bounds [0.05, 0.99]
    defined within the current execution layer.
    """
    model = StochasticLGDModel(baseline_lgd=0.45)
    
    # Test extreme input boundaries to explicitly force mathematical domain clipping
    ltv_vector = np.array([0, 80, 500], dtype=np.int32)
    
    # The underlying module relies on legacy np.random.uniform. 
    # Global state seeding is mandatory here to prevent stochastic test flakiness.
    np.random.seed(101)
    lgd_vector = model.calculate_lgd_vector(ltv_vector)
    
    assert lgd_vector.shape == (3,)
    assert lgd_vector.dtype == np.float64
    
    # Verify empirical outputs match the implemented limit operators.
    # A loan with 0 LTV yields a mapped mean of 0.10. A uniform shock of -0.15
    # yields -0.05. The global clip MUST bind this correctly to 0.05.
    assert np.all(lgd_vector >= 0.05)
    assert np.all(lgd_vector <= 0.99)

def test_vasicek_monte_carlo_engine_execution(sample_simulation_chunk):
    """
    Executes the vectorized structural default framework. 
    Validates matrix dimensionality, strict bounds on allocated loss, 
    and mathematical invariance by manipulating the global PRNG pointer.
    """
    num_simulations = 1000
    num_sectors = 2
    
    # Generate globally deterministic macroeconomic scenarios
    np.random.seed(42)
    y_global = np.random.normal(0.0, 1.0, size=num_simulations).astype(np.float64)
    z_global = np.random.normal(0.0, 1.0, size=(num_sectors, num_simulations)).astype(np.float64)
    
    # Establish baseline legacy state to construct the first idiosyncratic epsilon matrix
    np.random.seed(999)
    portfolio_loss_distribution = VasicekMonteCarloEngine.execute_chunk_simulation(
        chunk=sample_simulation_chunk,
        y_global=y_global,
        z_global=z_global
    )
    
    # Vector dimensionality and contiguous byte alignments
    assert portfolio_loss_distribution.shape == (num_simulations,)
    assert portfolio_loss_distribution.dtype == np.float64
    
    # Total portfolio losses bounded [0, aggregate UPB]
    total_portfolio_upb = np.sum(sample_simulation_chunk.unpaid_principal_balance)
    assert np.all(portfolio_loss_distribution >= 0.0)
    assert np.all(portfolio_loss_distribution <= total_portfolio_upb)
    
    np.random.seed(999)
    identical_loss_distribution = VasicekMonteCarloEngine.execute_chunk_simulation(
        chunk=sample_simulation_chunk,
        y_global=y_global,
        z_global=z_global
    )
    
    np.testing.assert_array_equal(portfolio_loss_distribution, identical_loss_distribution)