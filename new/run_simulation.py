#!/usr/bin/env python3

import os
import sys
import subprocess

# Check if SUMO_HOME is set
if 'SUMO_HOME' not in os.environ:
    print("Please set SUMO_HOME environment variable")
    exit(1)

# Add SUMO tools to path
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

try:
    import traci
    import sumolib
except ImportError:
    print("Couldn't import SUMO modules. Make sure SUMO is installed correctly.")
    exit(1)

# Start SUMO with GUI
sumoBinary = sumolib.checkBinary('sumo-gui')
sumoCmd = [sumoBinary, "-c", "crossroads.sumocfg"]

print("Starting SUMO simulation...")
print("If a GUI doesn't appear, check that all XML files are correct and SUMO is properly installed.")

# Try to start the simulation
try:
    traci.start(sumoCmd)
    print("Simulation started successfully. View it in the SUMO GUI window.")
    
    # Run for 3600 steps (1 hour)
    for step in range(3600):
        traci.simulationStep()
        
        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Simulation step: {step}")
    
    # Close the simulation
    traci.close()
    print("Simulation completed!")
    
except Exception as e:
    print(f"Error running simulation: {e}")
    print("Try running SUMO directly with: sumo-gui -c crossroads.sumocfg")