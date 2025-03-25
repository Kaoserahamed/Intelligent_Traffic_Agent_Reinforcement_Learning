import os
import sys
import subprocess

def create_network():
    try:
        # Create nodes file
        with open('intersection.nod.xml', 'w') as f:
            f.write('''<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <node id="center" x="0.0" y="0.0" type="traffic_light"/>
    <node id="north" x="0.0" y="500.0" type="priority"/>
    <node id="south" x="0.0" y="-500.0" type="priority"/>
    <node id="east" x="500.0" y="0.0" type="priority"/>
    <node id="west" x="-500.0" y="0.0" type="priority"/>
</nodes>''')

        # Create edges file
        with open('intersection.edg.xml', 'w') as f:
            f.write('''<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <edge id="north_to_center" from="north" to="center" priority="1" numLanes="2" speed="13.89"/>
    <edge id="center_to_north" from="center" to="north" priority="1" numLanes="2" speed="13.89"/>
    
    <edge id="south_to_center" from="south" to="center" priority="1" numLanes="2" speed="13.89"/>
    <edge id="center_to_south" from="center" to="south" priority="1" numLanes="2" speed="13.89"/>
    
    <edge id="east_to_center" from="east" to="center" priority="1" numLanes="2" speed="13.89"/>
    <edge id="center_to_east" from="center" to="east" priority="1" numLanes="2" speed="13.89"/>
    
    <edge id="west_to_center" from="west" to="center" priority="1" numLanes="2" speed="13.89"/>
    <edge id="center_to_west" from="center" to="west" priority="1" numLanes="2" speed="13.89"/>
</edges>''')

        # Determine netconvert path
        if sys.platform.startswith('win'):
            netconvert_path = os.path.join(os.getenv('SUMO_HOME', ''), 'bin', 'netconvert.exe')
        else:
            netconvert_path = 'netconvert'

        # Generate network with comprehensive error handling
        result = subprocess.run(
            [netconvert_path, 
             '-n', 'intersection.nod.xml', 
             '-e', 'intersection.edg.xml', 
             '-o', 'intersection.net.xml'],
            capture_output=True,
            text=True
        )

        # Check for errors
        if result.returncode != 0:
            print("Network generation failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise Exception("Network conversion failed")

        print("Network generated successfully!")

    except Exception as e:
        print(f"Error in network generation: {e}")
        raise

    finally:
        # Clean up temporary files
        try:
            if os.path.exists('intersection.nod.xml'):
                os.remove('intersection.nod.xml')
            if os.path.exists('intersection.edg.xml'):
                os.remove('intersection.edg.xml')
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

def main():
    try:
        create_network()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()