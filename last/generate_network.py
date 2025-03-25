import os
import subprocess
import xml.etree.ElementTree as ET

def count_traffic_light_links(net_file):
    """
    Parse the network file to count the number of traffic light links
    """
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        # Find traffic light nodes
        tl_nodes = [node for node in root.findall('.//node') if node.get('type') == 'traffic_light']
        
        if not tl_nodes:
            print("No traffic light nodes found!")
            return 0
        
        # Assuming the first traffic light node
        tl_node_id = tl_nodes[0].get('id')
        
        # Count total links for this traffic light
        links = [edge for edge in root.findall('.//edge') 
                 if edge.get('from') == tl_node_id or edge.get('to') == tl_node_id]
        
        num_links = len(links)
        
        print(f"Detected {num_links} traffic light links")
        return num_links
    
    except Exception as e:
        print(f"Error parsing network file: {e}")
        return 0

def create_nodes_xml():
    nodes_content = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <node id="1" x="0.0" y="0.0" type="traffic_light"/>
    <node id="2" x="100.0" y="0.0" type="dead_end"/>
    <node id="3" x="0.0" y="100.0" type="dead_end"/>
    <node id="4" x="-100.0" y="0.0" type="dead_end"/>
    <node id="5" x="0.0" y="-100.0" type="dead_end"/>
</nodes>
"""
    with open("nodes.xml", "w") as f:
        f.write(nodes_content)

def create_edges_xml():
    edges_content = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <edge id="2to1" from="2" to="1" priority="1" numLanes="2" speed="13.89"/>
    <edge id="1to2" from="1" to="2" priority="1" numLanes="2" speed="13.89"/>
    
    <edge id="3to1" from="3" to="1" priority="1" numLanes="2" speed="13.89"/>
    <edge id="1to3" from="1" to="3" priority="1" numLanes="2" speed="13.89"/>
    
    <edge id="4to1" from="4" to="1" priority="1" numLanes="2" speed="13.89"/>
    <edge id="1to4" from="1" to="4" priority="1" numLanes="2" speed="13.89"/>
    
    <edge id="5to1" from="5" to="1" priority="1" numLanes="2" speed="13.89"/>
    <edge id="1to5" from="1" to="5" priority="1" numLanes="2" speed="13.89"/>
</edges>
"""
    with open("edges.xml", "w") as f:
        f.write(edges_content)

def generate_network():
    # Create nodes and edges XML files
    create_nodes_xml()
    create_edges_xml()

    # Use netconvert with additional parameters
    try:
        subprocess.run([
            "netconvert", 
            "--node-files=nodes.xml", 
            "--edge-files=edges.xml", 
            "--output-file=intersection.net.xml",
            "--default.speed=13.89"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Network generation error: {e}")
    
    # Optional: clean up temporary files
    os.remove("nodes.xml")
    os.remove("edges.xml")

def create_route_file():
    routes_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="50" guiShape="passenger"/>
    <vType id="truck" accel="1.0" decel="4.0" sigma="0.5" length="15" minGap="2.5" maxSpeed="25" guiShape="truck"/>

    <route id="east_route" edges="2to1 1to2"/>
    <route id="west_route" edges="4to1 1to4"/>
    <route id="north_route" edges="3to1 1to3"/>
    <route id="south_route" edges="5to1 1to5"/>

    <flow id="east_flow" type="car" route="east_route" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="west_flow" type="car" route="west_route" begin="0" end="3600" vehsPerHour="300"/>
    <flow id="north_flow" type="truck" route="north_route" begin="0" end="3600" vehsPerHour="200"/>
    <flow id="south_flow" type="truck" route="south_route" begin="0" end="3600" vehsPerHour="200"/>
</routes>
"""
    with open("intersection.rou.xml", "w") as f:
        f.write(routes_content)

def create_additional_file(num_links):
    # Ensure num_links is even and greater than 0
    if num_links <= 0 or num_links % 2 != 0:
        print("Invalid number of links for traffic light logic.")
        return
    
    # Calculate the number of connections (each link has 4 connections)
    num_connections = num_links * 4
    
    # Generate state strings
    half_connections = num_connections // 2
    green_state = 'GGGG' * (half_connections // 4) + 'rrrr' * (half_connections // 4)
    yellow_state = 'yyyy' * (num_connections // 4)
    
    additional_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    <tlLogic id="1" type="static" programID="1" offset="0">
        <phase duration="31" state="{green_state}"/>
        <phase duration="6"  state="{yellow_state}"/>
        <phase duration="31" state="{green_state[::-1]}"/>
        <phase duration="6"  state="{yellow_state}"/>
    </tlLogic>
</additional>
"""
    with open("intersection.add.xml", "w") as f:
        f.write(additional_content)

def create_config_file():
    config_content = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="intersection.rou.xml"/>
        <additional-files value="intersection.add.xml"/>
    </input>
    
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
    
    <processing>
        <ignore-route-errors value="true"/>
    </processing>
</configuration>
"""
    with open("intersection.sumocfg", "w") as f:
        f.write(config_content)

def main():
    generate_network()
    create_route_file()
    
    # Count links and create additional file
    num_links = count_traffic_light_links("intersection.net.xml")
    create_additional_file(num_links)
    
    create_config_file()
    print("Simulation files generated successfully!")
    if num_links > 0: # Only create config if additional file was created
        create_config_file()
        print("Simulation files generated successfully!")
    else:
        print("Simulation file generation aborted due to errors.")

if __name__ == "__main__":
    main()