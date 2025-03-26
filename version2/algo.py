import os
import sys
import traci
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
        """
        self.sumo_config = sumo_config
        
        # Traffic light parameters
        self.traffic_light_id = "1"
        
        # Define phases explicitly based on the additional XML
        # Green phases with a total length of 20 characters each
        self.phases = [
            "GGGgrrrrrrrrrrrr",  # North-South green
            "rrrrGGGgrrrrrrrr",  # East-West green
            "rrrrrrrrGGGgrrrr",  # Diagonal top-left to bottom-right green
            "rrrrrrrrrrrrGGGg"   # Diagonal bottom-left to top-right green
        ]

        # Yellow phases with a total length of 20 characters each
        self.yellow_phases = [
            "yyyyrrrrrrrrrrrr",  # Yellow for North-South
            "rrrryyyyrrrrrrrr",  # Yellow for East-West
            "rrrrrrrryyyyrrrr",  # Yellow for Diagonal 1
            "rrrrrrrrrrrryyyy"   # Yellow for Diagonal 2
        ]

        # All red phase with a total length of 20 characters
        self.all_red_phase = "rrrrrrrrrrrrrrrr"  # All red phase
        
        # Phase duration parameters
        self.min_phase_duration = 42  # Green phase duration
        self.max_phase_duration = 48  # Including yellow and all-red phases
        
        # Lanes for each phase
        self.phase_lanes = {
            0: ["1to3_0", "3to1_0"],     # North-South green
            1: ["1to2_0", "2to1_0"],     # East-West green
            2: ["1to4_0", "4to1_0"],     # Diagonal top-left to bottom-right
            3: ["1to5_0", "5to1_0"]      # Diagonal bottom-left to top-right
        }
        
        # Current phase tracking
        self.current_phase = None
        self.current_phase_duration = 0
    
    def count_waiting_vehicles(self, traci_instance):
        """
        Count waiting vehicles in lanes for each phase
        
        :param traci_instance: TraCI connection
        :return: Dictionary of waiting vehicle counts per phase
        """
        waiting_counts = {}
        
        for phase, lanes in self.phase_lanes.items():
            waiting_count = 0
            for lane in lanes:
                try:
                    # Count vehicles with speed below 0.1 m/s (considered waiting)
                    waiting_vehicles = [
                        vid for vid in traci_instance.lane.getLastStepVehicleIDs(lane)
                        if traci_instance.vehicle.getSpeed(vid) < 0.1
                    ]
                    waiting_count += len(waiting_vehicles)
                except traci.exceptions.TraCIException:
                    print(f"Warning: Could not count waiting vehicles in lane {lane}")
            
            waiting_counts[phase] = waiting_count
        
        return waiting_counts
    
    def check_emergency_vehicles(self, traci_instance):
        """
        Check for emergency vehicles in all lanes
        
        :param traci_instance: TraCI connection
        :return: List of phases with emergency vehicles
        """
        emergency_phases = []
        
        for phase, lanes in self.phase_lanes.items():
            for lane in lanes:
                try:
                    # Get vehicles in the lane
                    vehicle_ids = traci_instance.lane.getLastStepVehicleIDs(lane)
                    
                    # Check if any vehicle is an emergency vehicle
                    for vehicle_id in vehicle_ids:
                        try:
                            vehicle_type = traci_instance.vehicle.getVehicleClass(vehicle_id)
                            if vehicle_type == "emergency":
                                emergency_phases.append(phase)
                                break
                        except traci.exceptions.TraCIException:
                            continue
                except traci.exceptions.TraCIException:
                    print(f"Warning: Could not check lane {lane}")
        
        return emergency_phases
    
    def select_next_phase(self, traci_instance, current_phase):
        """
        Select the next phase based on emergency vehicles and waiting vehicles
        
        :param traci_instance: TraCI connection
        :param current_phase: Current active phase
        :return: Next phase to activate
        """
        # Check emergency vehicles first
        emergency_phases = self.check_emergency_vehicles(traci_instance)
        
        # If emergency vehicles exist, prioritize them
        if emergency_phases:
            return emergency_phases[0]
        
        # Count waiting vehicles
        waiting_counts = self.count_waiting_vehicles(traci_instance)
        
        # Sort phases by waiting vehicle count in descending order
        sorted_phases = sorted(waiting_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return the phase with the most waiting vehicles
        next_phase = sorted_phases[0][0]
        
        # Ensure we don't repeat the current phase immediately
        if next_phase == current_phase and len(sorted_phases) > 1:
            next_phase = sorted_phases[1][0]
        
        return next_phase
    
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
            self.current_phase = random.randint(0, len(self.phases) - 1)
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
                        
                        # Select next phase based on emergency vehicles or waiting vehicles
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