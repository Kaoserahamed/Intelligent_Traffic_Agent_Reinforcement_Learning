import os
import sys
import subprocess
import time
import socket
import random
import traci
import sumolib

# Ensure SUMO tools are in the Python path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    print("Please declare environment variable 'SUMO_HOME'")
    sys.exit(1)

def find_free_port():
    """Find a free port to use for TraCI connection"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

class TrafficLightController:
    def __init__(self, tls_id):
        self.tls_id = tls_id
        self.directions = None
        self.current_green_direction = None
        self.traffic_light_phases = {
            'north': "GGggrrrrr",
            'south': "rrrrGGggr",
            'east':  "rrrrrrrGG",
            'west':  "GGrrrrrr"
        }

    def initialize_lanes(self):
        try:
            # Dynamically retrieve lane IDs
            all_lanes = traci.lane.getIDList()
            
            self.directions = {
                'north': [lane for lane in all_lanes if lane.startswith('3to1')],
                'south': [lane for lane in all_lanes if lane.startswith('5to1')],
                'east':  [lane for lane in all_lanes if lane.startswith('2to1')],
                'west':  [lane for lane in all_lanes if lane.startswith('4to1')]
            }
            
            # Print detected lanes
            for direction, lanes in self.directions.items():
                print(f"{direction.capitalize()} Lanes: {lanes}")
            
            return True
        except Exception as e:
            print(f"Error initializing lanes: {e}")
            return False

    def set_traffic_light_state(self, direction):
        try:
            # Validate direction
            if direction not in self.traffic_light_phases:
                print(f"Invalid direction: {direction}")
                return False

            # Get the corresponding state for the direction
            new_state = self.traffic_light_phases[direction]

            # Set the traffic light state
            traci.trafficlight.setRedYellowGreenState(self.tls_id, new_state)
            print(f"Set traffic light to {direction} direction: {new_state}")
            self.current_green_direction = direction
            return True

        except Exception as e:
            print(f"Error setting traffic light state: {e}")
            return False

    def count_vehicles_in_direction(self, direction):
        if not self.directions or direction not in self.directions:
            print(f"No lanes found for direction: {direction}")
            return 0

        vehicle_count = 0
        for lane in self.directions[direction]:
            try:
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                vehicle_count += len(vehicles)
            except Exception as e:
                print(f"Error counting vehicles on lane {lane}: {e}")
        return vehicle_count

    def adaptive_traffic_control(self):
        # Determine the direction with most vehicles
        vehicle_counts = {
            direction: self.count_vehicles_in_direction(direction)
            for direction in self.directions.keys()
        }
        
        # Print vehicle counts for debugging
        print("Vehicle Counts:", vehicle_counts)

        # Find direction with max vehicles
        most_congested = max(vehicle_counts, key=vehicle_counts.get)
        
        # If different from current green direction, change
        if most_congested != self.current_green_direction:
            print(f"Changing to {most_congested} direction")
            self.set_traffic_light_state(most_congested)

def start_sumo_process(config_path, port):
    """
    Start SUMO as a separate process with specified TraCI port
    """
    try:
        # Prepare SUMO command
        sumoBinary = sumolib.checkBinary('sumo-gui')
        
        # Construct SUMO command with explicit TraCI port
        sumo_cmd = [
            sumoBinary, 
            "-c", config_path,  
            "--start", 
            "--quit-on-end",
            "--no-step-log",
            "--duration-log.disable",
            f"--remote-port={port}"
        ]
        
        # Start SUMO process
        process = subprocess.Popen(sumo_cmd)
        return process
    except Exception as e:
        print(f"Error starting SUMO process: {e}")
        return None

def connect_to_traci(port, max_attempts=10):
    """
    Attempt to connect to TraCI server with retries
    """
    for attempt in range(max_attempts):
        try:
            traci.init(port)
            print(f"Successfully connected to TraCI on port {port}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1}: Could not connect to TraCI server: {e}")
            time.sleep(1)
    return False

def run_simulation():
    # Construct full path to configuration file
    config_path = os.path.join(os.getcwd(), "intersection.sumocfg")
    
    # Verify config file exists
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return

    # Find a free port
    port = find_free_port()
    sumo_process = None

    try:
        # Start SUMO process
        sumo_process = start_sumo_process(config_path, port)
        if not sumo_process:
            print("Failed to start SUMO process")
            return

        # Connect to TraCI
        if not connect_to_traci(port):
            print("Failed to connect to TraCI")
            return

        # Get traffic light IDs
        tls_ids = traci.trafficlight.getIDList()
        print("Traffic Light IDs:", tls_ids)

        if not tls_ids:
            print("No traffic lights found in the simulation!")
            return

        # Initialize traffic light controller
        tls_controller = TrafficLightController(tls_ids[0])
        
        # Initialize lanes
        if not tls_controller.initialize_lanes():
            print("Failed to initialize lanes!")
            return

        # Initial green light direction
        if not tls_controller.set_traffic_light_state('north'):
            print("Failed to set initial traffic light state!")
            return

        # Simulation loop
        simulation_duration = 7200  # 2 hours
        step = 0
        while traci.simulation.getTime() < simulation_duration:
            try:
                traci.simulationStep()
                
                # Every 30 seconds, check and adapt traffic light
                if step % 30 == 0:
                    tls_controller.adaptive_traffic_control()
                
                step += 1

            except Exception as step_error:
                print(f"Error in simulation step {step}: {step_error}")
                break

    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        # Ensure SUMO is closed
        try:
            traci.close()
        except Exception:
            pass
        
        # Terminate SUMO process if it's still running
        if sumo_process:
            sumo_process.terminate()

def main():
    run_simulation()

if __name__ == "__main__":
    main()