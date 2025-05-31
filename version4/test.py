import traci

sumo_cmd = [
    "sumo-gui",
    "-c", "intersection.sumocfg",
    "--start",
    "--remote-port", "50311"
]

try:
    traci.start(sumo_cmd)
    print("Connected to SUMO!")
    traci.close()
except Exception as e:
    print(f"Error connecting: {e}")
