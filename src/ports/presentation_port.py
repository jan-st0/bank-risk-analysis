from typing import Protocol

import numpy as np

class RenderVisualPresentation(Protocol):
    def render_loss_distribution_histogram(self, losses: np.ndarray, output_directory: str) -> None:
        ...