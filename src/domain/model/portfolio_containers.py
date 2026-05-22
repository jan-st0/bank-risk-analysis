from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import TypeAlias

npFloat: TypeAlias = npt.NDArray[np.float64]
npInt: TypeAlias = npt.NDArray[np.int32]
npIntSmall: TypeAlias = npt.NDArray[np.int8]
npStr: TypeAlias = npt.NDArray[np.bytes_]

@dataclass(slots=True)
class PortfolioClientRaw:
    """
    Entity class for communication with repository
    It contains multiple client records
    """
    ubp: npInt
    property_type: npIntSmall
    delinq: npIntSmall
    zero_bal_code: npIntSmall
    credit_score: npIntSmall
    ltv_ratio: npIntSmall
    debt_to_inc: npIntSmall
    ocp_status: npIntSmall
    loan_age: npIntSmall
    loan_id: npInt




@dataclass(slots=True)
class PortfolioClientsSim:
    """
    contains only the columns necessary for core simulation
    """
    loan_vector: npFloat
    sector_vector: npIntSmall



@dataclass(slots=True)
class SectorCoeffs:
    """
    coefficients for each secotor
    in case of single family loan dataset the sectors are property types
    """

    condominium_vector: npFloat

    co_operative_vector: npFloat

    planned_urban_development_vector: npFloat

    manufactured_home_vector: npFloat

    single_family_home_vector: npFloat

def int_to_sector(sector: int, source: SectorCoeffs) -> npFloat:
    match sector:
        case 0:
            return source.condominium_vector
        case 1:
            return source.co_operative_vector
        case 2:
            return source.planned_urban_development_vector
        case 3:
            return source.manufactured_home_vector
        case 4:
            return source.single_family_home_vector
        case _:
            raise ValueError("Invalid sector int mapping argument")