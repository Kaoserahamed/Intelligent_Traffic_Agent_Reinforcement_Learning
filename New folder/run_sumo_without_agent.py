import os
import sys
import traci

# Set the path to SUMO tools (make sure SUMO_HOME is defined)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set the SUMO_HOME environment variable.")

def run_sumo_simulation(sumo_cfg_path, sumo_binary_path, max_steps=1000):
    sumo_cmd = [
        sumo_binary_path,
        "-c", sumo_cfg_path,
        "--start",
        "--no-step-log",
        "--no-warnings"
    ]

    print(f"Starting SUMO simulation with config: {sumo_cfg_path}")

    try:
        traci.start(sumo_cmd)
        step = 0
        while step < max_steps:
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("No more vehicles in simulation.")
                break

            traci.simulationStep()

            # Print basic simulation info
            vehicle_ids = traci.vehicle.getIDList()
            print(f"Step {step}: Active vehicles: {len(vehicle_ids)}")

            step += 1

        print("Simulation finished.")

    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        traci.close()

if __name__ == "__main__":
    # Update these paths based on your setup
    sumo_cfg = "intersection.sumocfg"
    sumo_binary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"

    run_sumo_simulation(sumo_cfg, sumo_binary, max_steps=1000)
