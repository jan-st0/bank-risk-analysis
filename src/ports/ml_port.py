from typing import Protocol
import pandas as pd
from domain.model.portfolio_containers import TrainingBatch, npFloat64

class ModelTrainingPort(Protocol):
    def train_and_calibrate(self, batch: TrainingBatch, save_dir: str) -> dict[str, float]:
        ...

class ProbabilityOfDefaultInferencePort(Protocol):
    def load_model_asset(self, save_dir: str) -> None:
        """Loads serialized model assets to prepare the worker context."""
        ...

    def predict_probabilities_direct(self, underlying_features: pd.DataFrame) -> npFloat64:
        """Returns empirical probability arrays using direct feature processing."""
        ...