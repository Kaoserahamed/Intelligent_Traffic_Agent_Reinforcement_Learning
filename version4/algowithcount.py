import traci
import os
import sys
from collections import defaultdict

# Import SUMO library
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Define the incoming edges we want to monitor
INCOMING_EDGES = {
    'East': '2to1',   # vehicles coming from east
    'North': '3to1',  # vehicles coming from north
    'West': '4to1',   # vehicles coming from west
    'South': '5to1'   # vehicles coming from south
}

# Maps directions to their corresponding traffic light phases
DIRECTION_TO_PHASE = {
    'North': 0,  # First phase: North-South green
    'East': 3,   # Second phase: East-West green
    'South': 6,  # Third phase: South-North green
    'West': 9    # Fourth phase: West-East green
}

# Traffic light parameters
BASE_DURATION = 10  # Minimum green time for each direction
MAX_DURATION = 300   # Maximum green time for each direction
YELLOW_DURATION = 3 # Yellow phase duration
ALL_RED_DURATION = 3 # All-red phase duration
WEIGHT_PER_VEHICLE = 2 # Additional seconds per waiting vehicle

def get_edge_vehicle_count(edge_id):
    """Count vehicles on an edge"""
    return traci.edge.getLastStepVehicleNumber(edge_id)

def calculate_phase_duration(vehicle_count):
    """Calculate the green phase duration based on vehicle count"""
    # Base duration plus additional time based on waiting vehicles
    duration = BASE_DURATION + (vehicle_count * WEIGHT_PER_VEHICLE)
    # Cap at maximum duration
    return min(duration, MAX_DURATION)

def run():
    # Start SUMO with the given configuration
    traci.start(["sumo-gui", "-c", "intersection.sumocfg"])
    
    step = 0
    current_direction_index = 0
    directions = list(INCOMING_EDGES.keys())
    phase_time_left = 0
    current_phase_type = "red"  # Can be "green", "yellow", "red"
    
    while step < 3600:  # Run for 1 hour simulation time
        traci.simulationStep()
        
        # Dictionary to store vehicle counts
        counts = {}
        
        # Count vehicles on each incoming edge
        for direction, edge in INCOMING_EDGES.items():
            counts[direction] = get_edge_vehicle_count(edge)
        
        # Clear the console (Windows)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Print the counts and time allocation in a formatted way
        print(f"\nTime step: {step}")
        print("=" * 50)
        print("Direction\tVehicle Count\tAllocated Time")
        print("-" * 50)
        
        # If we need to calculate a new phase
        if phase_time_left <= 0:
            if current_phase_type == "green":
                # Switch to yellow
                current_phase_type = "yellow"
                phase_time_left = YELLOW_DURATION
                traci.trafficlight.setPhase('1', DIRECTION_TO_PHASE[directions[current_direction_index]] + 1)
            elif current_phase_type == "yellow":
                # Switch to all red
                current_phase_type = "red"
                phase_time_left = ALL_RED_DURATION
                traci.trafficlight.setPhase('1', DIRECTION_TO_PHASE[directions[current_direction_index]] + 2)
            elif current_phase_type == "red":
                # Move to the next direction
                current_direction_index = (current_direction_index + 1) % len(directions)
                # Switch to green
                current_phase_type = "green"
                # Calculate the duration based on waiting vehicles
                current_direction = directions[current_direction_index]
                green_duration = calculate_phase_duration(counts[current_direction])
                phase_time_left = green_duration
                traci.trafficlight.setPhase('1', DIRECTION_TO_PHASE[current_direction])
        
        # Print information for each direction
        for i, direction in enumerate(directions):
            count = counts[direction]
            if i == current_direction_index and current_phase_type == "green":
                status = f"GREEN ({phase_time_left}s left)"
                allocated_time = calculate_phase_duration(count)
            elif i == current_direction_index and current_phase_type == "yellow":
                status = f"YELLOW ({phase_time_left}s left)"
                allocated_time = 0  # Not currently allocating green time
            else:
                status = "WAITING"
                allocated_time = calculate_phase_duration(count)  # What it would get when its turn comes
            
            print(f"{direction:<12}\t{count:<14}\t{allocated_time}s ({status})")
        
        print("=" * 50)
        
        # Decrement the phase timer
        phase_time_left -= 1
        
        step += 1
    
    traci.close()

if __name__ == "__main__":
    run()