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
        "_unpaid_principal_balance", "_base_probabilities", "_ltv_vector"
    )

    def __init__(
        self, upb: npFloat64, pd_vector: npFloat64, ltv: npInt32
    ):
        self._unpaid_principal_balance = np.ascontiguousarray(upb, dtype=np.float64)
        self._base_probabilities = np.ascontiguousarray(pd_vector, dtype=np.float64)
        self._ltv_vector = np.ascontiguousarray(ltv, dtype=np.int32)

    @property
    def unpaid_principal_balance(self) -> npFloat64: return self._unpaid_principal_balance
    
    @property
    def base_probabilities(self) -> npFloat64: return self._base_probabilities
    
    @property
    def ltv_vector(self) -> npInt32: return self._ltv_vector

    def __len__(self) -> int: return self._unpaid_principal_balance.shape[0]