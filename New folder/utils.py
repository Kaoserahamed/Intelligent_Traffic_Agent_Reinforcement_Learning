# utils.py

import pandas as pd
import traci

def count_waiting_vehicles(traci_instance, phase_lanes):
    waiting_counts = {}
    for phase, lanes in phase_lanes.items():
        count = 0
        for lane in lanes:
            try:
                vehicle_ids = traci_instance.lane.getLastStepVehicleIDs(lane)
                count += sum(1 for vid in vehicle_ids if traci_instance.vehicle.getSpeed(vid) < 0.1)
            except traci.exceptions.TraCIException:
                print(f"Could not access lane: {lane}")
        waiting_counts[phase] = count
    return waiting_counts

def log_traffic_data(traci_instance, step, vehicle_wait_times, lane_data_log):
    all_lanes = set()
    for lane_id in traci_instance.lane.getIDList():
        all_lanes.add(lane_id)

    step_data = {"step": step, "stats": {}}

    for lane in all_lanes:
        try:
            vehicle_ids = traci_instance.lane.getLastStepVehicleIDs(lane)
            total_wait = 0
            for vid in vehicle_ids:
                speed = traci_instance.vehicle.getSpeed(vid)
                if speed < 0.1:
                    vehicle_wait_times[vid] = vehicle_wait_times.get(vid, 0) + 1
                total_wait += vehicle_wait_times.get(vid, 0)
            step_data["stats"][lane] = {
                "num_vehicles": len(vehicle_ids),
                "total_waiting_time": total_wait
            }
        except:
            continue

    lane_data_log.append(step_data)

def export_to_excel(lane_data_log, filename="traffic_data.xlsx"):
    records = []
    for entry in lane_data_log:
        for lane, stats in entry["stats"].items():
            records.append({
                "step": entry["step"],
                "lane": lane,
                "num_vehicles": stats["num_vehicles"],
                "total_waiting_time": stats["total_waiting_time"]
            })
    df = pd.DataFrame(records)
    df.to_excel(filename, index=False)

# Add this to utils.py
import matplotlib.pyplot as plt

def plot_performance(excel_file):
    df = pd.read_excel(excel_file)

    # Check the actual columns:
    print(df.columns)

    # Use the actual columns:
    grouped = df.groupby("Step").agg({
        "TotalWaitingTime": "sum",
        # You don't have "num_vehicles" directly, but you have lane counts like "Lane_0_Count", "Lane_1_Count", ...
        # So sum across these columns:
    }).reset_index()

    # Sum all lane count columns per step
    lane_count_cols = [col for col in df.columns if col.endswith("_Count")]
    grouped["TotalVehicles"] = df.groupby("Step")[lane_count_cols].sum().sum(axis=1)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(grouped["Step"], grouped["TotalWaitingTime"], label="Total Waiting Time")
    plt.plot(grouped["Step"], grouped["TotalVehicles"], label="Total Vehicles")
    plt.xlabel("Simulation Step")
    plt.ylabel("Count")
    plt.title(f"Traffic Performance: {excel_file}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
