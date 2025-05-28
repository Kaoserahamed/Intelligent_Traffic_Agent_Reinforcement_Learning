import traci
import os
import sys
from collections import defaultdict
import math

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
BASE_DURATION = 10  # Minimum green time for each direction
MAX_DURATION = 450   # Maximum green time for each direction
YELLOW_DURATION = 3 # Yellow phase duration
ALL_RED_DURATION = 3 # All-red phase duration

# Parameters for progressive weighting
BASE_WEIGHT = 1.0   # Weight for first vehicle
WEIGHT_INCREMENT = 2.9  # How much additional weight each subsequent vehicle adds
MAX_WEIGHT_MULTIPLIER = 9.0  # Maximum multiplier for the weight

# Traffic jam threshold - consider this many vehicles to be a jam
TRAFFIC_JAM_THRESHOLD = 10

def get_edge_vehicle_count(edge_id):
    """Count vehicles on an edge - includes approaching vehicles"""
    # Count vehicles currently on the edge
    count = traci.edge.getLastStepVehicleNumber(edge_id)
    
    # Also count vehicles waiting to enter the edge (provides more accurate count)
    waiting_count = traci.edge.getLastStepHaltingNumber(edge_id)
    
    return count

def get_enhanced_vehicle_count(edge_id):
    """Get enhanced vehicle count including vehicles in the queue"""
    # Basic count on the edge
    count = traci.edge.getLastStepVehicleNumber(edge_id)
    
    # Try to detect additional vehicles that might be waiting to enter
    # This looks at the actual queue length on the edge
    queue_length = traci.edge.getLastStepLength(edge_id)
    estimated_vehicles = max(count, int(queue_length / 7.5))  # Assuming average vehicle length + gap
    
    return estimated_vehicles

def calculate_phase_duration(vehicle_count):
    """
    Calculate the green phase duration based on vehicle count
    with progressive weighting to account for queue length
    """
    if vehicle_count == 0:
        return BASE_DURATION
    
    # Calculate progressive weight factor
    # First vehicles get base weight, later vehicles get progressively higher weights
    total_weight = 0
    for i in range(vehicle_count):
        # Progressively increase weight for vehicles further back in the queue
        weight = min(BASE_WEIGHT + (i * WEIGHT_INCREMENT), BASE_WEIGHT * MAX_WEIGHT_MULTIPLIER)
        total_weight += weight
    
    # Base duration plus additional time based on weighted vehicle count
    duration = BASE_DURATION + (total_weight * 1.5)  # 1.5 seconds per weight unit
    
    # Cap at maximum duration
    return min(round(duration), MAX_DURATION)

def run():
    # Start SUMO with the given configuration
    traci.start(["sumo-gui", "-c", "intersection.sumocfg"])
    
    step = 0
    current_direction_index = 0  # Start with the first direction in the cycle
    phase_time_left = 0
    current_phase_type = "red"  # Can be "green", "yellow", "red"
    
    # Keep track of total waiting time for each direction
    cumulative_wait = {direction: 0 for direction in DIRECTION_ORDER}
    
    # Track emergency mode for handling traffic jams
    emergency_mode = False
    emergency_direction = None
    
    while step < 3600:  # Run for 1 hour simulation time
        traci.simulationStep()
        
        # Dictionary to store vehicle counts
        counts = {}
        enhanced_counts = {}
        
        # Count vehicles on each incoming edge and update waiting times
        for direction, edge in INCOMING_EDGES.items():
            counts[direction] = get_edge_vehicle_count(edge)
            enhanced_counts[direction] = get_enhanced_vehicle_count(edge)
            
            # Increment wait time for all directions except current green
            if not (current_phase_type == "green" and DIRECTION_ORDER[current_direction_index] == direction):
                cumulative_wait[direction] += counts[direction]
        
        # Check for traffic jams that need emergency handling
        jam_directions = []
        for direction, count in enhanced_counts.items():
            if count >= TRAFFIC_JAM_THRESHOLD:
                jam_directions.append((direction, count))
        
        # Clear the console (Windows/Linux)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Print the counts and time allocation in a formatted way
        print(f"\nTime step: {step}")
        print("=" * 70)
        print("Direction\tVehicle Count\tCumulative Wait\tAllocated Time\tStatus")
        print("-" * 70)
        
        # If we need to calculate a new phase
        if phase_time_left <= 0:
            if current_phase_type == "green":
                # Reset cumulative wait for the direction that just had green
                cumulative_wait[DIRECTION_ORDER[current_direction_index]] = 0
                # Switch to yellow
                current_phase_type = "yellow"
                phase_time_left = YELLOW_DURATION
                traci.trafficlight.setPhase('1', DIRECTION_TO_PHASE[DIRECTION_ORDER[current_direction_index]] + 1)
            elif current_phase_type == "yellow":
                # Switch to all red
                current_phase_type = "red"
                phase_time_left = ALL_RED_DURATION
                traci.trafficlight.setPhase('1', DIRECTION_TO_PHASE[DIRECTION_ORDER[current_direction_index]] + 2)
            elif current_phase_type == "red":
                # Check if we should enter emergency mode for a traffic jam
                if jam_directions and not emergency_mode:
                    # Sort jammed directions by vehicle count
                    jam_directions.sort(key=lambda x: x[1], reverse=True)
                    worst_jam_direction, worst_jam_count = jam_directions[0]
                    
                    # If the jam is severe enough and not the next direction in the cycle
                    next_direction = DIRECTION_ORDER[(current_direction_index + 1) % len(DIRECTION_ORDER)]
                    if worst_jam_count >= TRAFFIC_JAM_THRESHOLD * 1.5 and worst_jam_direction != next_direction:
                        emergency_mode = True
                        emergency_direction = worst_jam_direction
                        # Find index of emergency direction
                        for i, direction in enumerate(DIRECTION_ORDER):
                            if direction == emergency_direction:
                                current_direction_index = i
                                break
                else:
                    # If we were in emergency mode, return to normal cycling
                    emergency_mode = False
                    emergency_direction = None
                    # Move to the next direction in the fixed cycle if not in emergency mode
                    current_direction_index = (current_direction_index + 1) % len(DIRECTION_ORDER)
                
                # Switch to green
                current_phase_type = "green"
                # Calculate the duration based on waiting vehicles with progressive weighting
                current_direction = DIRECTION_ORDER[current_direction_index]
                
                # In case of emergency, give maximum time to clear the jam
                if emergency_mode and current_direction == emergency_direction:
                    green_duration = MAX_DURATION
                else:
                    green_duration = calculate_phase_duration(counts[current_direction])
                
                phase_time_left = green_duration
                traci.trafficlight.setPhase('1', DIRECTION_TO_PHASE[current_direction])
        
        # Print information for each direction
        for i, direction in enumerate(DIRECTION_ORDER):
            count = counts[direction]
            enhanced_count = enhanced_counts[direction]
            wait = cumulative_wait[direction]
            
            # If there's a major discrepancy between counts, use the enhanced count
            if enhanced_count > count * 1.5:
                display_count = f"{count} ({enhanced_count} detected)"
            else:
                display_count = str(count)
            
            if i == current_direction_index and current_phase_type == "green":
                if emergency_mode and direction == emergency_direction:
                    status = f"EMERGENCY GREEN ({phase_time_left}s left)"
                else:
                    status = f"GREEN ({phase_time_left}s left)"
                allocated_time = calculate_phase_duration(count)
                if emergency_mode and direction == emergency_direction:
                    allocated_time = MAX_DURATION
            elif i == current_direction_index and current_phase_type == "yellow":
                status = f"YELLOW ({phase_time_left}s left)"
                allocated_time = 0  # Not currently allocating green time
            elif i == current_direction_index and current_phase_type == "red":
                status = f"ALL RED ({phase_time_left}s left)"
                allocated_time = 0
            else:
                next_in_cycle = (current_direction_index + 1) % len(DIRECTION_ORDER) == i
                if next_in_cycle and not (emergency_mode and emergency_direction != direction):
                    status = "NEXT IN CYCLE"
                else:
                    status = "WAITING"
                allocated_time = calculate_phase_duration(count)  # What it would get when its turn comes
            
            print(f"{direction:<12}\t{display_count:<14}\t{wait:<15}\t{allocated_time}s\t\t{status}")
        
        # Special indicator for traffic jam conditions
        for direction, count in enhanced_counts.items():
            if count >= TRAFFIC_JAM_THRESHOLD:
                print(f"⚠️ TRAFFIC JAM ALERT: {count} vehicles waiting on {direction} side")
        
        if emergency_mode:
            print(f"⚠️ EMERGENCY MODE ACTIVE: Prioritizing {emergency_direction} direction to clear traffic jam")
            
        print("=" * 70)
        
        # Decrement the phase timer
        phase_time_left -= 1
        
        step += 1
    
    traci.close()

if __name__ == "__main__":
    run()