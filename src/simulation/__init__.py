"""
Simulation package: Simulator and Terminal Dashboard.
"""

from .simulator import (
    SimulationScenario,
    DynamicTarget,
    TelemetryFrame,
    AutopilotSimulator,
)
from .dashboard import TerminalDashboard

__all__ = [
    "SimulationScenario",
    "DynamicTarget",
    "TelemetryFrame",
    "AutopilotSimulator",
    "TerminalDashboard",
]
