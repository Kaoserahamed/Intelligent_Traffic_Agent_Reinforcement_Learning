import os
import sys
import traci
import numpy as np
import random

# Ensure SUMO tools are in the Python path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise ValueError("SUMO_HOME environment variable not set")

class SumoTrafficRL:
    def __init__(self, sumo_config):
        """
        Initialize RL Traffic Light Controller for SUMO simulation
        
        :param sumo_config: SUMO configuration file
        """
        self.sumo_config = sumo_config
        
        # Traffic light parameters
        self.traffic_light_id = "1"  # From your XML configuration
        
        # Define phases explicitly
        self.phases = [
            "GGgrrrrrrrrr",  # North green
            "rrrGGgrrrrrr",  # East green
            "rrrrrrGGgrrr",  # Diagonal top-left to bottom-right green
            "rrrrrrrrrGGg"   # Diagonal bottom-left to top-right green
        ]
        
        # Yellow and all-red phases corresponding to green phases
        self.yellow_phases = [
            "yyyrrrrrrrrr",  # Yellow for North
            "rrryyyrrrrrr",  # Yellow for East
            "rrrrrryyyrrr",  # Yellow for Diagonal 1
            "rrrrrrrrryyy"   # Yellow for Diagonal 2
        ]
        self.all_red_phase = "rrrrrrrrrrrr"
        
        # Phase duration parameters
        self.min_phase_duration = 120  # 2 minutes (120 simulation steps)
        self.max_phase_duration = 300  # 5 minutes (300 simulation steps)
        
        # Lanes for each phase
        self.phase_lanes = {
            0: ["5to1_0", "3to1_0"],    # North-South
            1: ["2to1_0", "4to1_0"],    # East-West
            2: ["5to2_0", "2to3_0"],    # Diagonal top-left to bottom-right
            3: ["5to4_0", "4to3_0"]     # Diagonal bottom-left to top-right
        }
        
        # Current phase tracking
        self.current_phase = None
        self.current_phase_duration = 0
    
    def check_emergency_vehicles(self, traci_instance):
        """
        Check for emergency vehicles in all lanes
        
        :param traci_instance: TraCI connection
        :return: List of phases with emergency vehicles, sorted by priority
        """
        emergency_phases = []
        
        for phase, lanes in self.phase_lanes.items():
            for lane in lanes:
                try:
                    # Get vehicles in the lane
                    vehicle_ids = traci_instance.lane.getLastStepVehicleIDs(lane)
                    
                    # Check if any vehicle is an emergency vehicle
                    for vehicle_id in vehicle_ids:
                        vehicle_type = traci_instance.vehicle.getVehicleClass(vehicle_id)
                        if vehicle_type == "emergency":
                            emergency_phases.append(phase)
                            break
                except traci.exceptions.TraCIException:
                    print(f"Warning: Could not check lane {lane}")
        
        return emergency_phases
    
    def count_vehicles_in_lanes(self, traci_instance):
        """
        Count vehicles in lanes for each phase
        
        :param traci_instance: TraCI connection
        :return: Dictionary of vehicle counts per phase
        """
        vehicle_counts = {}
        
        for phase, lanes in self.phase_lanes.items():
            vehicle_count = 0
            for lane in lanes:
                try:
                    vehicle_count += traci_instance.lane.getLastStepVehicleNumber(lane)
                except traci.exceptions.TraCIException:
                    print(f"Warning: Could not count vehicles in lane {lane}")
            
            vehicle_counts[phase] = vehicle_count
        
        return vehicle_counts
    
    def select_next_phase(self, traci_instance, current_phase):
        """
        Select the next phase based on emergency vehicles and vehicle counts
        
        :param traci_instance: TraCI connection
        :param current_phase: Current active phase
        :return: Next phase to activate
        """
        # Check emergency vehicles first
        emergency_phases = self.check_emergency_vehicles(traci_instance)
        
        # If emergency vehicles exist, prioritize them
        if emergency_phases:
            return emergency_phases[0]
        
        # If no emergency vehicles, count vehicles in lanes
        vehicle_counts = self.count_vehicles_in_lanes(traci_instance)
        
        # Sort phases by vehicle count in descending order
        sorted_phases = sorted(vehicle_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return the phase with the most vehicles
        return sorted_phases[0][0]
    
    def set_traffic_light_phase(self, traci_instance, action):
        """
        Set traffic light phase with yellow and all-red transitions
        
        :param traci_instance: TraCI connection
        :param action: Selected action
        """
        try:
            # Set yellow phase
            traci_instance.trafficlight.setRedYellowGreenState(
                self.traffic_light_id, 
                self.yellow_phases[action]
            )
            traci_instance.simulationStep()
            
            # Set all-red phase
            traci_instance.trafficlight.setRedYellowGreenState(
                self.traffic_light_id, 
                self.all_red_phase
            )
            traci_instance.simulationStep()
            
            # Set green phase
            traci_instance.trafficlight.setRedYellowGreenState(
                self.traffic_light_id, 
                self.phases[action]
            )
        except traci.exceptions.TraCIException as e:
            print(f"Error setting traffic light phase: {e}")
    
    def run_simulation(self, num_episodes=100, max_steps=3600):
        """
        Run SUMO simulation with advanced traffic light control
        
        :param num_episodes: Number of simulation episodes
        :param max_steps: Maximum steps per episode
        """
        sumo_binary = "sumo-gui"  # For visualization
        
        for episode in range(num_episodes):
            # Randomly select initial phase
            self.current_phase = random.randint(0, 3)
            self.current_phase_duration = 0
            
            # Launch SUMO
            sumo_cmd = [
                sumo_binary,
                "-c", self.sumo_config,
                "--start",
                "--no-step-log", 
                "--no-warnings"
            ]
            
            try:
                # Start TraCI
                traci.start(sumo_cmd)
                
                # Simulation loop
                step = 0
                
                while step < max_steps:
                    # Check if simulation is still running
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        print("No more vehicles in simulation")
                        break
                    
                    # Perform simulation step
                    traci.simulationStep()
                    
                    # Increment phase duration
                    self.current_phase_duration += 1
                    
                    # Check phase change conditions
                    if (self.current_phase_duration >= self.min_phase_duration and 
                        self.current_phase_duration <= self.max_phase_duration):
                        
                        # Select next phase based on emergency vehicles or vehicle counts
                        next_phase = self.select_next_phase(traci, self.current_phase)
                        
                        # Change phase if needed
                        if next_phase != self.current_phase:
                            self.set_traffic_light_phase(traci, next_phase)
                            self.current_phase = next_phase
                            self.current_phase_duration = 0
                    
                    # Force phase change after max duration
                    elif self.current_phase_duration > self.max_phase_duration:
                        next_phase = self.select_next_phase(traci, self.current_phase)
                        self.set_traffic_light_phase(traci, next_phase)
                        self.current_phase = next_phase
                        self.current_phase_duration = 0
                    
                    step += 1
                
                print(f"Episode {episode + 1} completed.")
            
            except traci.exceptions.TraCIException as e:
                print(f"TraCI Exception: {e}")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
            finally:
                # Ensure TraCI is closed
                if traci.isLoaded():
                    traci.close()

# Main execution
def main():
    # Path to your SUMO configuration file
    sumo_config = "intersection.sumocfg"
    
    try:
        # Initialize and run traffic controller
        traffic_controller = SumoTrafficRL(sumo_config)
        traffic_controller.run_simulation(
            num_episodes=10,  # Reduced for demonstration
            max_steps=3600    # 1 hour simulation
        )
        
        print("Simulation completed successfully.")
    
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()