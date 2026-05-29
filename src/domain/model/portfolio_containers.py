import numpy as np
import pandas as pd
import numpy.typing as npt
from dataclasses import dataclass
from typing import TypeAlias

npFloat64: TypeAlias = npt.NDArray[np.float64]
npInt32: TypeAlias = npt.NDArray[np.int32]

@dataclass(frozen=True)
class TrainingBatch:
    features: pd.DataFrame
    targets: npInt32
    feature_names: list[str]

class PortfolioSimulationChunk:
    __slots__ = (
        "_unpaid_principal_balance", "_sector_indices", "_base_probabilities",
        "_beta_vector", "_rho_vector", "_ltv_vector"
    )

    def __init__(
        self, upb: npFloat64, sectors: npInt32, pd_vector: npFloat64,
        beta: npFloat64, rho: npFloat64, ltv: npInt32
    ):
        self._unpaid_principal_balance = np.ascontiguousarray(upb, dtype=np.float64)
        self._sector_indices = np.ascontiguousarray(sectors, dtype=np.int32)
        self._base_probabilities = np.ascontiguousarray(pd_vector, dtype=np.float64)
        self._beta_vector = np.ascontiguousarray(beta, dtype=np.float64)
        self._rho_vector = np.ascontiguousarray(rho, dtype=np.float64)
        self._ltv_vector = np.ascontiguousarray(ltv, dtype=np.int32)

    @property
    def unpaid_principal_balance(self) -> npFloat64: return self._unpaid_principal_balance
    @property
    def sector_indices(self) -> npInt32: return self._sector_indices
    @property
    def base_probabilities(self) -> npFloat64: return self._base_probabilities
    @property
    def beta_vector(self) -> npFloat64: return self._beta_vector
    @property
    def rho_vector(self) -> npFloat64: return self._rho_vector
    @property
    def ltv_vector(self) -> npInt32: return self._ltv_vector

    def __len__(self) -> int: return self._unpaid_principal_balance.shape[0]