import sumolib
import traci

def analyze_traffic_simulation(simulation_file):
    # Initialize SUMO connection
    traci.start(["sumo", "-c", simulation_file])

    # Simulation analysis variables
    vehicle_counts = {
        'passenger_car': 0,
        'emergency_ambulance': 0,
        'police_car': 0,
        'truck': 0
    }

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            # Step through simulation
            traci.simulationStep()

            # Count vehicle types
            for veh_id in traci.vehicle.getIDList():
                veh_type = traci.vehicle.getVehicleClass(veh_id)
                if veh_type in vehicle_counts:
                    vehicle_counts[veh_type] += 1

    finally:
        # Print analysis results
        print("Vehicle Type Counts:")
        for veh_type, count in vehicle_counts.items():
            print(f"{veh_type}: {count}")

        traci.close()

# Run the analysis
analyze_traffic_simulation("traffic_simulation.sumocfg")