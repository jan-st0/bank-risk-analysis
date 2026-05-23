import numpy as np
import numpy.typing as npt
from typing import TypeAlias

npFloat: TypeAlias = npt.NDArray[np.float64]
npInt32: TypeAlias = npt.NDArray[np.int32]
npInt8: TypeAlias = npt.NDArray[np.int8]
StructuredArray: TypeAlias = npt.NDArray[np.void]

PORTFOLIO_STRUCTURED_DTYPE = np.dtype([
    ("current_actual_upb", np.int32),
    ("borrower_credit_score", np.int8),
    ("ltv_ratio", np.int8),
    ("debt_to_income", np.int8),
    ("loan_identifier", np.int32),
    ("split", np.float64),
    ("property_type", np.int8),
    ("current_loan_delinquency_status", np.int8),
    ("zero_balance_code", np.int8),
    ("occupancy_status", np.int8),
    ("loan_age", np.int8),
    ("loan_report_date", np.int32),
    ("sector_id", np.int32)
])

class PortfolioClientRaw:
    """
    Entity class for communication with repository
    It contains multiple client records
    """
    __slots__ = ("_data",)

    def __init__(self, structured_array: StructuredArray):
        if structured_array.dtype != PORTFOLIO_STRUCTURED_DTYPE:
            raise TypeError(
                f"Memory layout mismatch. Expected {PORTFOLIO_STRUCTURED_DTYPE}, got {structured_array.dtype}"
            )
        self._data = structured_array

    @property
    def data(self) -> StructuredArray:
        return self._data

    @property
    def upb_vector(self) -> npInt32:
        return self._data["current_actual_upb"]

    @property
    def credit_score_vector(self) -> npInt8:
        return self._data["borrower_credit_score"]
    
    @property
    def sector_vector(self) -> npInt32:
        return self._data["sector_id"]
    

    def __len__(self) -> int:
        return self._data.shape[0]

class PortfolioClientsSim:
    """
    contains only the columns necessary for core simulation
    """
    __slots__ = ("_loan_vector", "_sector_vector")
    def __init__(self, loan_vector: npFloat, sector_vector: npInt32):
        if loan_vector.ndim != 1 or sector_vector.ndim != 1:
            raise ValueError("Simulation vectors must be strictly 1-dimensional.")
        if loan_vector.shape[0] != sector_vector.shape[0]:
            raise ValueError(
                f"Vector alignment mismatch. loan_vector ({loan_vector.shape[0]}) "
                f"must match sector_vector ({sector_vector.shape[0]})."
            )
        
        self._loan_vector = loan_vector
        self._sector_vector = sector_vector 
    
    @property
    def loan_vector(self) -> npFloat:
        return self._loan_vector
    
    @property
    def sector_vector(self) -> npInt32:
        return self._sector_vector
    
    def __len__(self):
        return self._loan_vector.shape[0]

class SectorCoeff:
    """
    Each client has a set of coefficiets based on its sector/property type association.
    These coefficients tells how sensitive the client is on change of a global factor, like economy.
    """
    __slots__ = ("_matrix",)

    def __init__(self, coeff_matrix: npFloat):
        if coeff_matrix.ndim != 2:
            raise ValueError("Coefficient space must be structured as a 2D matrix")
        self._matrix = np.ascontiguousarray(coeff_matrix)
    
    @property
    def matrix(self) -> npFloat:
        return self._matrix
    
    def get_coefficients_for_portfolio(self, sector_indicies: npInt32) -> npFloat:
        return self._matrix[sector_indicies]