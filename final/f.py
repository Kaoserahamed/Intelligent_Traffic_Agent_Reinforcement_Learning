import os
import sys
import subprocess

def create_net_file():
    # Create nodes file
    with open("intersection.nod.xml", "w") as nodes:
        nodes.write('''<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <node id="center" x="0.0" y="0.0" type="traffic_light"/>
    <node id="N" x="0.0" y="100.0"/>
    <node id="S" x="0.0" y="-100.0"/>
    <node id="E" x="100.0" y="0.0"/>
    <node id="W" x="-100.0" y="0.0"/>
</nodes>
''')

    # Create edges file
    with open("intersection.edg.xml", "w") as edges:
        edges.write('''<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <edge from="N" to="center" id="north_in" priority="1" numLanes="1" speed="13.89"/>
    <edge from="center" to="S" id="north_out" priority="1" numLanes="1" speed="13.89"/>
    
    <edge from="S" to="center" id="south_in" priority="1" numLanes="1" speed="13.89"/>
    <edge from="center" to="N" id="south_out" priority="1" numLanes="1" speed="13.89"/>
    
    <edge from="E" to="center" id="east_in" priority="1" numLanes="1" speed="13.89"/>
    <edge from="center" to="W" id="east_out" priority="1" numLanes="1" speed="13.89"/>
    
    <edge from="W" to="center" id="west_in" priority="1" numLanes="1" speed="13.89"/>
    <edge from="center" to="E" id="west_out" priority="1" numLanes="1" speed="13.89"/>
</edges>
''')

    # Generate network
    subprocess.call([
        "netconvert", 
        "-n", "intersection.nod.xml", 
        "-e", "intersection.edg.xml", 
        "-o", "intersection.net.xml",
        "--no-internal-links", "false"
    ])

def create_route_file():
    with open("intersection.rou.xml", "w") as routes:
        routes.write('''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="1.5" decel="4.5" length="5" minGap="2" maxSpeed="16.67"/>
    
    <route id="route_north" edges="north_in north_out"/>
    <route id="route_south" edges="south_in south_out"/>
    <route id="route_east" edges="east_in east_out"/>
    <route id="route_west" edges="west_in west_out"/>
    
    <vehicle id="car_north1" type="car" route="route_north" depart="0" departPos="0"/>
    <vehicle id="car_north2" type="car" route="route_north" depart="5" departPos="0"/>
    <vehicle id="car_south1" type="car" route="route_south" depart="2" departPos="0"/>
    <vehicle id="car_east1" type="car" route="route_east" depart="1" departPos="0"/>
    <vehicle id="car_west1" type="car" route="route_west" depart="3" departPos="0"/>
</routes>
''')

def create_additional_file():
    with open("intersection.add.xml", "w") as add:
        add.write('''<?xml version="1.0" encoding="UTF-8"?>
<additional>
    <tlLogic id="center" type="static" programID="1" offset="0">
        <!-- North Straight -->
        <phase duration="42" state="GGgrrrGGgrrrGGgrrrGGgrr"/>
        <phase duration="3"  state="yyyrrryyyrrryyyrrryyyrrr"/>
        <phase duration="3"  state="rrrrrrrrrrrrrrrrrrrrrrrr"/>
        
        <!-- North Right Turn -->
        <phase duration="42" state="GGGrrrGGGrrrGGGrrrGGGrrr"/>
        <phase duration="3"  state="yyyrrryyyrrryyyrrryyyrrr"/>
        <phase duration="3"  state="rrrrrrrrrrrrrrrrrrrrrrrr"/>
        
        <!-- East Straight -->
        <phase duration="42" state="rrrGGgrrrGGgrrrGGgrrrGGg"/>
        <phase duration="3"  state="rrryyyrrryyyrrryyyrrryyy"/>
        <phase duration="3"  state="rrrrrrrrrrrrrrrrrrrrrrrr"/>
        
        <!-- East Right Turn -->
        <phase duration="42" state="rrrGGGrrrGGGrrrGGGrrrGGG"/>
        <phase duration="3"  state="rrryyyrrryyyrrryyyrrryyy"/>
        <phase duration="3"  state="rrrrrrrrrrrrrrrrrrrrrrrr"/>
    </tlLogic>
</additional>
''')

def create_sumo_config():
    with open("intersection.sumocfg", "w") as cfg:
        cfg.write('''<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="intersection.rou.xml"/>
        <additional-files value="intersection.add.xml"/>
    </input>
    <time-to-teleport value="-1"/>
    <time>
        <begin value="0"/>
        <end value="300"/>
    </time>
</configuration>
''')

def run_simulation():
    import traci
    sumoBinary = "sumo-gui"
    traci.start([sumoBinary, "-c", "intersection.sumocfg"])
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
    
    traci.close()

if __name__ == "__main__":
    create_net_file()
    create_route_file()
    create_additional_file()
    create_sumo_config()
    run_simulation()