import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import TypeAlias

npFloat64: TypeAlias = npt.NDArray[np.float64]
npInt32: TypeAlias = npt.NDArray[np.int32]

@dataclass(frozen=True)
class TrainingBatch:
    """
    Pure domain container representing a processed training payload.
    Completely isolated from storage technology, schemas, and byte-string structures.
    """
    features: npFloat64
    targets: npInt32
    feature_names: list[str]

class PortfolioSimulationChunk:
    """
    Encapsulates a sub-segment of the simulation portfolio, optimized for
    high-throughput vectorized math operations.
    """
    __slots__ = ("_unpaid_principal_balance", "_sector_indices", "_base_probabilities")

    def __init__(self, upb: npFloat64, sectors: npInt32, pd_vector: npFloat64):
        if upb.ndim != 1 or sectors.ndim != 1 or pd_vector.ndim != 1:
            raise ValueError("All multidimensional metrics must be simplified to contiguous 1D spaces.")
        if not (upb.shape[0] == sectors.shape[0] == pd_vector.shape[0]):
            raise ValueError("Structural dimension alignment mismatch across chunk parameters.")
        
        self._unpaid_principal_balance = np.ascontiguousarray(upb, dtype=np.float64)
        self._sector_indices = np.ascontiguousarray(sectors, dtype=np.int32)
        self._base_probabilities = np.ascontiguousarray(pd_vector, dtype=np.float64)

    @property
    def unpaid_principal_balance(self) -> npFloat64:
        return self._unpaid_principal_balance

    @property
    def sector_indices(self) -> npInt32:
        return self._sector_indices

    @property
    def base_probabilities(self) -> npFloat64:
        return self._base_probabilities

    def __len__(self) -> int:
        return self._unpaid_principal_balance.shape[0]