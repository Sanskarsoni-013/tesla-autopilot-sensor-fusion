#!/usr/bin/env python3
"""
Example autonomous driving simulation run script.
Executes scenario and outputs telemetry data.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.simulation.simulator import AutopilotSimulator, SimulationScenario
from src.simulation.dashboard import TerminalDashboard


def run_demo():
    print("=" * 80)
    print("  TESLA AUTOPILOT SENSOR INTELLIGENCE - SCENARIO DEMO RUNNER")
    print("=" * 80)

    for scenario in [SimulationScenario.HIGHWAY_CRUISE, SimulationScenario.EMERGENCY_AEB_STOP]:
        print(f"\n---> Starting scenario: {scenario.value.upper()} (20 steps)")
        sim = AutopilotSimulator(scenario=scenario)
        for i in range(20):
            frame = sim.step()
            if i % 5 == 0 or frame.threat.threat_level.value != "normal":
                hud = TerminalDashboard.render_frame(frame)
                print(hud)
                print()

    print("\n[✔] Demo scenario execution completed successfully.")


if __name__ == "__main__":
    run_demo()
