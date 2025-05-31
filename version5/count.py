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

# Define the edges we want to monitor
EDGES = ['1to2', '1to3', '1to4', '1to5', '2to1', '3to1', '4to1', '5to1']

def run():
    # Start SUMO with the given configuration
    traci.start(["sumo-gui", "-c", "intersection.sumocfg"])
    
    step = 0
    
    while step < 3600:  # Run for 1 hour simulation time
        traci.simulationStep()
        
        # Dictionary to store vehicle counts
        vehicle_counts = defaultdict(int)
        waiting_counts = defaultdict(int)
        
        # Count vehicles on each edge
        for edge in EDGES:
            # Get all vehicles on the edge
            vehicles = traci.edge.getLastStepVehicleIDs(edge)
            vehicle_counts[edge] = len(vehicles)
            
            # Count waiting vehicles (speed < 0.1 m/s)
            waiting_vehicles = 0
            for vehicle in vehicles:
                if traci.vehicle.getSpeed(vehicle) < 0.1:
                    waiting_vehicles += 1
            waiting_counts[edge] = waiting_vehicles
        
        # Print the counts
        print(f"\nTime step: {step}")
        print("Edge\t\tTotal Vehicles\tWaiting Vehicles")
        print("-" * 50)
        for edge in EDGES:
            print(f"{edge}\t\t{vehicle_counts[edge]}\t\t{waiting_counts[edge]}")
        
        step += 1
    
    traci.close()

if __name__ == "__main__":
    run()