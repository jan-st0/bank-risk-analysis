import os
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from ports.database_port import ClientRepository
from ports.ml_port import ProbabilityOfDefaultInferencePort
from domain.model.portfolio_containers import PortfolioSimulationChunk
from domain.model.math_engine import VasicekMonteCarloEngine

class CreditPortfolioSimulationEngine:
    def __init__(self, db_repo: ClientRepository, inference_adapter: ProbabilityOfDefaultInferencePort, save_dir: str):
        self._db_repo = db_repo
        self._inference_adapter = inference_adapter
        self._save_dir = save_dir

    def ensure_model_exists(self, training_orchestrator) -> None:
        model_path = os.path.join(self._save_dir, "calibrated_pd_xgboost.pkl")
        if not os.path.exists(model_path):
            print(f"[SYSTEM] Essential inference asset not found at {model_path}.")
            print(f"[SYSTEM] Initiating mathematical training and calibration pipeline dynamically...")
            training_orchestrator.execute_training_pipeline()

    def run_portfolio_simulation(self, chunk_size: int, num_simulations: int, num_workers: int) -> None:
        self._inference_adapter.load_model_asset(self._save_dir)
        print(f"[ENGINE] Distributing {num_simulations} Monte Carlo paths across {num_workers} parallel workers...")
        
        prng = np.random.Generator(np.random.PCG64(42))
        y_global = prng.standard_normal(num_simulations, dtype=np.float64)
        z_global = prng.standard_normal((100, num_simulations), dtype=np.float64)
        
        total_loss_distribution = np.zeros(num_simulations, dtype=np.float64)
        t0 = time.time()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for chunk_idx, chunk_df in enumerate(self._db_repo.stream_simulation_chunks(chunk_size=chunk_size)):
                pd_vector = self._inference_adapter.predict_probabilities_direct(chunk_df)
                
                sim_chunk = PortfolioSimulationChunk(
                    upb=chunk_df["current_actual_upb"].to_numpy(dtype=np.float64),
                    sectors=chunk_df["sector_id"].to_numpy(dtype=np.int32),
                    pd_vector=pd_vector,
                    beta=chunk_df["beta_economy"].to_numpy(dtype=np.float64),
                    rho=chunk_df["asset_correlation"].to_numpy(dtype=np.float64),
                    ltv=chunk_df["ltv_ratio"].to_numpy(dtype=np.int32)
                )
                
                chunk_seed_seq = np.random.SeedSequence([42, chunk_idx])
                
                futures.append(
                    executor.submit(
                        VasicekMonteCarloEngine.execute_chunk_simulation,
                        sim_chunk, y_global, z_global, chunk_seed_seq
                    )
                )
            
            for future in futures:
                total_loss_distribution += future.result()

        t1 = time.time()
        print(f"[ENGINE] Monte Carlo integrations completed in {t1 - t0:.2f} seconds.")
        self._generate_validation_report(total_loss_distribution, num_simulations)

    def _generate_validation_report(self, loss_distribution: np.ndarray, num_sims: int) -> None:
        expected_loss = np.mean(loss_distribution)
        var_999 = np.percentile(loss_distribution, 99.9)
        economic_capital = var_999 - expected_loss
        standard_error = np.std(loss_distribution, ddof=1) / np.sqrt(num_sims)
        
        print("\n=== SYSTEMIC RISK VALIDATION REPORT ===")
        print(f"Expected Loss (EL):          ${expected_loss:,.2f}")
        print(f"Value at Risk (VaR 99.9%):   ${var_999:,.2f}")
        print(f"Economic Capital (EC):       ${economic_capital:,.2f}")
        print(f"Monte Carlo Standard Error:  ${standard_error:,.2f}")
        print("=======================================\n")
        
        self._render_loss_pdf(loss_distribution, expected_loss, var_999)

    def _render_loss_pdf(self, loss_distribution: np.ndarray, el: float, var: float) -> None:
        plt.figure(figsize=(10, 6))
        plt.hist(loss_distribution, bins=100, color='steelblue', density=True, alpha=0.7)
        plt.axvline(el, color='black', linestyle='dashed', linewidth=2, label=f'EL: ${el:,.0f}')
        plt.axvline(var, color='darkred', linestyle='solid', linewidth=2, label=f'VaR 99.9%: ${var:,.0f}')
        
        plt.title("Portfolio Loss Probability Density Function (Structural Default Topology)", fontsize=12, fontweight='bold')
        plt.xlabel("Total Portfolio Loss ($)", fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self._save_dir, "loss_pdf_report.png"))
        print(f"[SYSTEM] Validation PDF topology rendered to {self._save_dir}/loss_pdf_report.png")