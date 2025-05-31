#!/usr/bin/env python

import os
import sys
import traci
import traci.constants as tc

# Check if SUMO_HOME environment variable is set
if 'SUMO_HOME' not in os.environ:
    sys.exit("Please set the SUMO_HOME environment variable")

# Add SUMO tools to Python path
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

# Define the configuration file
sumocfg = "intersection.sumocfg"

# Start SUMO with the TraCI interface
sumoCmd = [os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo'), 
           "-c", sumocfg, 
           "--tripinfo-output", "tripinfo.xml"]

# Start SUMO with GUI (optional)
sumoCmd = [os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui'), 
           "-c", sumocfg, 
           "--tripinfo-output", "tripinfo.xml"]

# Start the TraCI connection
traci.start(sumoCmd)

# Identify all the incoming edges to the intersection
# Based on your network file, these are the edges that end at junction 1
incoming_edges = ["5to1", "2to1", "3to1", "4to1"]

try:
    # Main simulation loop
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # Print the current simulation step
        print(f"\nSimulation Step: {step}")
        print("=" * 40)
        
        # For each incoming edge, count the waiting vehicles
        for edge in incoming_edges:
            # Get all vehicles on the edge
            vehicles = traci.edge.getLastStepVehicleIDs(edge)
            waiting_count = 0
            
            # Count vehicles that are waiting (speed near zero)
            for vehicle in vehicles:
                if traci.vehicle.getSpeed(vehicle) < 0.1:
                    waiting_count += 1
            
            # Get the total number of vehicles on the edge
            total_count = len(vehicles)
            
            # Print the counts for this edge
            print(f"Edge {edge}: {waiting_count} waiting vehicles out of {total_count} total")
            
            # Optional: Print vehicle types
            vehicle_types = {}
            for vehicle in vehicles:
                vtype = traci.vehicle.getTypeID(vehicle)
                if vtype in vehicle_types:
                    vehicle_types[vtype] += 1
                else:
                    vehicle_types[vtype] = 1
            
            if vehicle_types:
                print(f"  Vehicle types: {vehicle_types}")
        
        step += 1

except Exception as e:
    print(f"Error during simulation: {e}")
finally:
    # Close the TraCI connection
    traci.close()
    print("Simulation complete")