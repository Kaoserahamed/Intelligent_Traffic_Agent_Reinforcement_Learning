import traci
import os
import sys

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

# Define the fixed cyclic order of directions
DIRECTION_ORDER = ['North', 'East', 'South', 'West']

# Maps directions to their corresponding traffic light phases
DIRECTION_TO_PHASE = {
    'North': 0,  # First phase: North-South green
    'East': 3,   # Second phase: East-West green
    'South': 6,  # Third phase: South-North green
    'West': 9    # Fourth phase: West-East green
}

# Traffic light parameters
MIN_GREEN_TIME = 10   # Minimum green time for each direction
MAX_GREEN_TIME = 450   # Maximum green time for each direction
YELLOW_TIME = 3       # Yellow phase duration
ALL_RED_TIME = 3      # All-red phase duration

def get_vehicle_count(edge_id):
    """Count vehicles on an edge"""
    return traci.edge.getLastStepVehicleNumber(edge_id)

def calculate_green_time(vehicle_count):
    """Calculate green time based on vehicle count"""
    if vehicle_count == 0:
        return MIN_GREEN_TIME
    
    # Simple formula: base time plus 2 seconds per vehicle
    green_time = MIN_GREEN_TIME + (vehicle_count * 2)
    
    # Cap at maximum duration
    return min(green_time, MAX_GREEN_TIME)

def run():
    # Start SUMO with the given configuration
    traci.start(["sumo-gui", "-c", "intersection.sumocfg"])
    
    step = 0
    current_direction_index = 0  # Start with the first direction in the cycle
    phase_time_left = 0
    current_phase_type = "red"  # Can be "green", "yellow", "red"
    
    # Track waiting vehicles for each direction
    waiting_times = {direction: 0 for direction in DIRECTION_ORDER}
    
    while step < 3600:  # Run for 1 hour simulation time
        traci.simulationStep()
        
        # Count vehicles on each incoming edge
        counts = {}
        for direction, edge in INCOMING_EDGES.items():
            counts[direction] = get_vehicle_count(edge)
            
            # Increment wait time for all directions except current green
            if not (current_phase_type == "green" and DIRECTION_ORDER[current_direction_index] == direction):
                waiting_times[direction] += counts[direction]
        
        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Print the current status
        print(f"\nTime step: {step}")
        print("=" * 70)
        print("Direction\tVehicle Count\tWaiting Time\tAllocated Time\tStatus")
        print("-" * 70)
        
        # If we need to calculate a new phase
        if phase_time_left <= 0:
            if current_phase_type == "green":
                # Reset waiting time for the direction that just had green
                waiting_times[DIRECTION_ORDER[current_direction_index]] = 0
                # Switch to yellow
                current_phase_type = "yellow"
                phase_time_left = YELLOW_TIME
                traci.trafficlight.setPhase('1', DIRECTION_TO_PHASE[DIRECTION_ORDER[current_direction_index]] + 1)
            
            elif current_phase_type == "yellow":
                # Switch to all red
                current_phase_type = "red"
                phase_time_left = ALL_RED_TIME
                traci.trafficlight.setPhase('1', DIRECTION_TO_PHASE[DIRECTION_ORDER[current_direction_index]] + 2)
            
            elif current_phase_type == "red":
                # Move to the next direction in the fixed cycle
                current_direction_index = (current_direction_index + 1) % len(DIRECTION_ORDER)
                
                # Switch to green
                current_phase_type = "green"
                current_direction = DIRECTION_ORDER[current_direction_index]
                
                # Calculate the duration based on waiting vehicles
                green_time = calculate_green_time(counts[current_direction])
                phase_time_left = green_time
                
                traci.trafficlight.setPhase('1', DIRECTION_TO_PHASE[current_direction])
        
                    # Print information for each direction
        for i, direction in enumerate(DIRECTION_ORDER):
            count = counts[direction]
            wait = waiting_times[direction]
            
            if i == current_direction_index and current_phase_type == "green":
                status = f"GREEN ({phase_time_left}s left)"
                allocated_time = calculate_green_time(count)
            elif i == current_direction_index and current_phase_type == "yellow":
                status = f"YELLOW ({phase_time_left}s left)"
                allocated_time = 0
            elif i == current_direction_index and current_phase_type == "red":
                status = f"ALL RED ({phase_time_left}s left)"
                allocated_time = 0
            else:
                # Determine next direction in cycle - this is the bug fix
                next_direction_index = (current_direction_index + 1) % len(DIRECTION_ORDER)
                next_in_cycle = (i == next_direction_index)
                status = "NEXT IN CYCLE" if next_in_cycle else "WAITING"
                allocated_time = calculate_green_time(count)
            
            print(f"{direction:<12}\t{count:<14}\t{wait:<15}\t{allocated_time}s\t\t{status}")
        
        print("=" * 70)
        
        # Decrement the phase timer
        phase_time_left -= 1
        step += 1
    
    traci.close()

if __name__ == "__main__":
    run()