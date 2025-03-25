import os
import sys
import traci
import sumolib

class TrafficLightController:
    def __init__(self, tls_id):
        self.tls_id = tls_id
        self.directions = self.get_lane_directions()
        self.current_green_direction = None

    def get_lane_directions(self):
        # Dynamically retrieve lane IDs
        all_lanes = traci.lane.getIDList()
        print("Available Lanes:", all_lanes)

        # Group lanes by direction
        directions = {
            'north': [lane for lane in all_lanes if lane.startswith('3') or lane.endswith('3to1')],
            'south': [lane for lane in all_lanes if lane.startswith('5') or lane.endswith('5to1')],
            'east': [lane for lane in all_lanes if lane.startswith('2') or lane.endswith('2to1')],
            'west': [lane for lane in all_lanes if lane.startswith('4') or lane.endswith('4to1')]
        }

        # Print detected lanes for each direction
        for direction, lanes in directions.items():
            print(f"{direction.capitalize()} Lanes: {lanes}")

        return directions

    def set_traffic_light_state(self, direction):
        try:
            # Get current state details
            current_program = traci.trafficlight.getProgram(self.tls_id)
            num_phases = traci.trafficlight.getNumPrograms(self.tls_id)
            
            print(f"Current Program: {current_program}")
            print(f"Number of Phases: {num_phases}")

            # Determine state based on direction
            if direction == 'north':
                new_state = "GGggrrrrr"
            elif direction == 'south':
                new_state = "rrrrGGggr"
            elif direction == 'east':
                new_state = "rrrrrrrGG"
            elif direction == 'west':
                new_state = "GGrrrrrr"
            else:
                print(f"Invalid direction: {direction}")
                return

            # Set the new state
            traci.trafficlight.setRedYellowGreenState(self.tls_id, new_state)
            print(f"Set traffic light to {direction} direction: {new_state}")
            
            self.current_green_direction = direction

        except Exception as e:
            print(f"Error setting traffic light state: {e}")

    def count_vehicles_in_direction(self, direction):
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
            for direction in self.directions
        }
        
        # Find direction with max vehicles
        most_congested = max(vehicle_counts, key=vehicle_counts.get)
        
        # If different from current green direction, change
        if most_congested != self.current_green_direction:
            print(f"Changing to {most_congested} direction")
            print("Vehicle Counts:", vehicle_counts)
            self.set_traffic_light_state(most_congested)

def run_simulation():
    # Prepare SUMO command
    sumoBinary = sumolib.checkBinary('sumo-gui')
    
    # Construct full path to configuration file
    config_path = os.path.join(os.getcwd(), "intersection.sumocfg")
    
    # Verify config file exists
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return

    try:
        # Start SUMO
        traci.start([sumoBinary, 
                     "-c", config_path,  
                     "--start", 
                     "--quit-on-end"])

        # Get traffic light IDs
        tls_ids = traci.trafficlight.getIDList()
        print("Traffic Light IDs:", tls_ids)

        # Initialize traffic light controller
        # Use the first traffic light ID found
        tls_controller = TrafficLightController(tls_ids[0] if tls_ids else '1')
        
        # Initial green light direction
        tls_controller.set_traffic_light_state('north')

        # Simulation loop
        simulation_duration = 7200  # 2 hours
        step = 0
        while traci.simulation.getTime() < simulation_duration:
            traci.simulationStep()
            
            # Every 30 seconds, check and adapt traffic light
            if step % 30 == 0:
                tls_controller.adaptive_traffic_control()
            
            step += 1

    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        # Ensure SUMO is closed
        traci.close()

def main():
    run_simulation()

if __name__ == "__main__":
    main()