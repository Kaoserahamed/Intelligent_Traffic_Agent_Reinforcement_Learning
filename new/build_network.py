#!/usr/bin/env python3

import os
import sys
import subprocess

# Check if SUMO_HOME is set
if 'SUMO_HOME' not in os.environ:
    print("Please set SUMO_HOME environment variable")
    sys.exit(1)

# Add SUMO tools to path
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

# Construct the netconvert command
# Construct the netconvert command
netconvert_cmd = [
    'netconvert',
    '-n', 'crossroads.nod.xml',
    '-e', 'crossroads.edg.xml',
    '-x', 'crossroads.con.xml',
    '-t', 'crossroads.tll.xml',
    '-o', 'crossroads.net.xml',
    '--tls.guess', 'true',
    '--lefthand', 'true',  # Add this line to enable left-hand driving
    '--verbose', 'true'
]

# Run netconvert
try:
    print("Generating network file...")
    result = subprocess.run(netconvert_cmd, 
                             capture_output=True, 
                             text=True, 
                             check=True)
    
    print("Network file generated successfully.")
    
except subprocess.CalledProcessError as e:
    print("Error generating network:")
    print(e.stderr)
    sys.exit(1)

# Additional network validation
print("\n--- NETWORK VALIDATION ---")
try:
    import sumolib
    
    # Try to load the network
    net = sumolib.net.readNet('crossroads.net.xml')
    
    # Get traffic lights
    tls_list = net.getTrafficLights()
    
    print(f"Number of Traffic Lights: {len(tls_list)}")
    
    for tls in tls_list:
        print(f"\nTraffic Light ID: {tls.getID()}")
        
        # Get connections for this traffic light
        connections = net.getTrafficLightConnections(tls.getID())
        
        print(f"Number of Connections: {len(connections)}")
        
        # Print details of each connection
        for i, conn in enumerate(connections):
            print(f"  Connection {i}:")
            print(f"    From Edge: {conn[0].getID()}")
            print(f"    To Edge: {conn[1].getID()}")
    
except ImportError:
    print("Could not import sumolib for network validation")
except Exception as e:
    print(f"Error during network validation: {e}")