### store constants and run parameters

import numpy as np
from dataclasses import dataclass

## ! Physical constants

@dataclass(frozen=True)
class AtomicUnits:
    hbar: float = 1.0
    e: float = 1.0
    m_e: float = 1.0
    a0: float = 1.0
    Eh: float = 1.0

AU = AtomicUnits()

# * Conversions

@dataclass(frozen=True)
class AUConversions:
    time_au_to_s: float = 2.418884e-17
    energy_au_to_ev: float = 27.211386
    length_au_to_m: float = 5.291772e-11

CONV = AUConversions()

## ! parameter configs
@dataclass
class BasisConfig:
    nmax: int = 2

@dataclass
class PulseConfig:
    E0: float = 0.01
    omega: float = 0.375
    N_cycles: float = 15
    t0: float = 300.0
    phase: float = 0.0
    polarization: tuple[float,float,float] = (0.0, 0.0, 1.0)

    def T(self):
        return self.N_cycles * 2 * np.pi / self.omega
    
@dataclass
class TimeGridConfig:
    dt: float = 0.15
    t_start: float = 0.0
    t_end: float = 600.0

@dataclass
class RelaxationConfig:
    enabled: bool = True
    gamma_scale: float = 1.0     # multiplier for physical rates

# * main script import
@dataclass
class SimulationConfig:
    basis: BasisConfig
    pulse: PulseConfig
    time: TimeGridConfig
    relaxation: RelaxationConfig