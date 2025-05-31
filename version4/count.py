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

def get_edge_vehicle_count(edge_id):
    """Count vehicles on an edge"""
    return traci.edge.getLastStepVehicleNumber(edge_id)

def run():
    # Start SUMO with the given configuration
    traci.start(["sumo-gui", "-c", "intersection.sumocfg"])
    
    step = 0
    
    while step < 3600:  # Run for 1 hour simulation time
        traci.simulationStep()
        
        # Dictionary to store vehicle counts
        counts = {}
        
        # Count vehicles on each incoming edge
        for direction, edge in INCOMING_EDGES.items():
            counts[direction] = get_edge_vehicle_count(edge)
        
        # Clear the console (Windows)
        os.system('cls')
        
        # Print the counts in a formatted way
        print(f"\nTime step: {step}")
        print("=" * 40)
        print("Direction\tVehicle Count")
        print("-" * 40)
        for direction, count in counts.items():
            print(f"{direction:<12}\t{count}")
        print("=" * 40)
        
        step += 1
    
    traci.close()

if __name__ == "__main__":
    run()