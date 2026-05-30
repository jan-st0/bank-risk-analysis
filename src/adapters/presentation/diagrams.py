import os
from matplotlib import pyplot as plt
import numpy as np


class VisualPresenter:
    def render_loss_distribution_histogram(self, losses: np.ndarray, output_directory: str) -> None:
        """
        Presentation Layer Adapter handling rendering operations explicitly
        outside of core domain service models.
        """
        expected_loss = np.mean(losses)
        var_999 = np.percentile(losses, 99.9)
        economic_capital = var_999 - expected_loss
        
        print("\n=== STOCHASTIC RISK METRICS REPORT ===")
        print(f"Portfolio Expected Loss (EL):    ${expected_loss:,.2f}")
        print(f"Value at Risk (VaR 99.9%):       ${var_999:,.2f}")
        print(f"Required Economic Capital (EC):  ${economic_capital:,.2f}")
        print("======================================\n")

        plt.figure(figsize=(11, 6))
        plt.hist(losses, bins=100, color='crimson', edgecolor='black', alpha=0.6, density=True)
        plt.axvline(expected_loss, color='black', linestyle='--', linewidth=2, label=f'EL: ${expected_loss:,.0f}')
        plt.axvline(var_999, color='blue', linestyle='-', linewidth=2, label=f'VaR 99.9%: ${var_999:,.0f}')
        
        plt.title("Stochastic Loss Density Function", fontsize=12, fontweight='bold')
        plt.xlabel("Consolidated Loss Matrix Volume ($)", fontsize=10)
        plt.ylabel("Relative Probability Denominator", fontsize=10)
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        
        output_target = os.path.join(output_directory, "loss_pdf_report.png")
        plt.savefig(output_target)
        print(f"[VISUALIZATION] Metric distribution report written cleanly to: {output_target}")