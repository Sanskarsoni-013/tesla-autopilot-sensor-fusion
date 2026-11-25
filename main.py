"""
Tesla Autopilot & Sensor Intelligence System - CLI Runner & Visualizer.
"""

import argparse
import sys
import time

from src.simulation.simulator import AutopilotSimulator, SimulationScenario
from src.simulation.dashboard import TerminalDashboard


def main():
    parser = argparse.ArgumentParser(
        description="Tesla Autopilot & Sensor Intelligence Simulation Runner"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="highway_cruise",
        choices=["highway_cruise", "obstacle_avoidance", "emergency_aeb_stop"],
        help="Driving scenario to simulate"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="Number of simulation steps to run"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.04,
        help="Playback delay per step in seconds (0 for instantaneous)"
    )

    args = parser.parse_args()

    scenario_map = {
        "highway_cruise": SimulationScenario.HIGHWAY_CRUISE,
        "obstacle_avoidance": SimulationScenario.OBSTACLE_AVOIDANCE,
        "emergency_aeb_stop": SimulationScenario.EMERGENCY_AEB_STOP,
    }
    scenario = scenario_map[args.scenario]

    print(f"\n[+] Initializing Tesla Autopilot Simulation: {scenario.value.upper()}...")
    sim = AutopilotSimulator(scenario=scenario)

    for step in range(args.steps):
        frame = sim.step()
        hud = TerminalDashboard.render_frame(frame)
        print("\033[H\033[J" if sys.stdout.isatty() else "", end="")
        print(hud)

        if args.rate > 0:
            time.sleep(args.rate)

    print("\n[+] Simulation run completed successfully.")


if __name__ == "__main__":
    main()
