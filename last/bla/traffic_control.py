import os
import sys
import traci
import random

class TrafficSignalController:
    def __init__(self):
        # Traffic light ID
        self.traffic_light_id = "center"
        
        # Define routes with their specific phases
        self.routes = [
            {
                "name": "north",
                "edges": ["north_to_center", "center_to_north"],
                "phase": "GGrrrrrrr"  # North green
            },
            {
                "name": "east",
                "edges": ["east_to_center", "center_to_east"],
                "phase": "rrGGrrrrr"  # East green
            },
            {
                "name": "south",
                "edges": ["south_to_center", "center_to_south"],
                "phase": "rrrrGGrrr"  # South green
            },
            {
                "name": "west",
                "edges": ["west_to_center", "center_to_west"],
                "phase": "rrrrrrGGr"  # West green
            }
        ]
        
        # Current route index
        self.current_route_index = 0
        
        # Signal timing parameters
        self.min_green_time = 15
        self.max_green_time = 45
        self.current_green_time = 0
        
        # Vehicle detection parameters
        self.waiting_threshold = 3

    def get_waiting_vehicles(self, route):
        """
        Count waiting vehicles for a specific route
        """
        try:
            total_waiting = 0
            
            # Check waiting vehicles on all edges of the route
            for edge in route["edges"]:
                lanes = traci.edge.getLaneIDs(edge)
                total_waiting += sum(
                    traci.lane.getLastStepHaltingNumber(lane) 
                    for lane in lanes
                )
            
            return total_waiting
        except Exception as e:
            print(f"Error detecting vehicles for {route['name']}: {e}")
            return 0

    def should_switch_route(self, route):
        """
        Determine if route should be switched
        """
        waiting_vehicles = self.get_waiting_vehicles(route)
        
        # Switch conditions
        conditions = [
            self.current_green_time >= self.max_green_time,  # Max time reached
            (self.current_green_time >= self.min_green_time and waiting_vehicles == 0),  # No waiting vehicles
            (self.current_green_time >= self.min_green_time and waiting_vehicles <= self.waiting_threshold)  # Low traffic
        ]
        
        return any(conditions)

    def update_traffic_lights(self):
        """
        Update traffic light state for current route
        """
        try:
            # Get current route
            current_route = self.routes[self.current_route_index]
            
            # Set traffic light phase
            traci.trafficlight.setRedYellowGreenState(
                self.traffic_light_id, 
                current_route["phase"]
            )
            
            print(f"Green signal for {current_route['name']} direction")
        except Exception as e:
            print(f"Traffic light update error: {e}")

    def manage_signal_timing(self):
        """
        Manage signal timing and route switching
        """
        # Increment green time
        self.current_green_time += 1
        
        # Get current route
        current_route = self.routes[self.current_route_index]
        
        # Check if route should be switched
        if self.should_switch_route(current_route):
            # Reset green time
            self.current_green_time = 0
            
            # Move to next route
            self.current_route_index = (self.current_route_index + 1) % len(self.routes)
            
            # Update traffic light
            self.update_traffic_lights()

def run_simulation(duration=3600):
    # Initialize traffic signal controller
    controller = TrafficSignalController()
    
    # Initial traffic light setup
    controller.update_traffic_lights()
    
    step = 0
    while step < duration:
        # Advance simulation
        traci.simulationStep()
        
        # Manage signal timing
        controller.manage_signal_timing()
        
        step += 1
    
    traci.close()

def main():
    # Generate network first
    import generate_network
    generate_network.create_network()

    # SUMO configuration
    sumoBinary = "sumo-gui"  # Use sumo for headless
    sumoConfig = "config.sumocfg"
    
    try:
        traci.start([sumoBinary, "-c", sumoConfig])
        run_simulation()
    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if traci.isLoaded():
            traci.close()

if __name__ == "__main__":
    main()