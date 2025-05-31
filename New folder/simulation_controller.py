import os
import pandas as pd
import traci
from datetime import datetime

class SumoSimulation:
    def __init__(self, config, agent, reward_method="default", sumo_binary="sumo"):
        self.config = config
        self.agent = agent
        self.reward_method = reward_method
        self.sumo_binary = sumo_binary
        self.traffic_light_id = "1"
        self.data_log = []

    def count_waiting_vehicles(self):
        lane_ids = traci.trafficlight.getControlledLanes(self.traffic_light_id)
        lane_waiting_times = {}
        lane_vehicle_counts = {}

        for lane in lane_ids:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            waiting_time = sum([traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicle_ids])
            lane_waiting_times[lane] = waiting_time
            lane_vehicle_counts[lane] = len(vehicle_ids)

        return lane_vehicle_counts, lane_waiting_times

    def get_state(self):
        lane_counts, _ = self.count_waiting_vehicles()
        return list(lane_counts.values())

    def calculate_reward(self):
        _, waiting_times = self.count_waiting_vehicles()
        return -sum(waiting_times.values())

    def apply_action(self, action_index):
        """Maps action index to traffic signal phase."""
        phase_map = {
            0: 0,  # NS Green
            1: 2,  # EW Green
            2: 4,  # Turn or custom
            3: 6   # Optional
        }
        try:
            phase = phase_map.get(action_index, 0)
            traci.trafficlight.setPhase(self.traffic_light_id, phase)
        except Exception as e:
            print(f"Error setting phase: {e}")

    from sumolib import checkBinary

    def run(self, num_episodes, max_steps, output_excel):

        direction_lanes = {
            "North": ["5to1_0", "5to1_1"],
            "East":  ["2to1_0", "2to1_1"],
            "South": ["3to1_0", "3to1_1"],
            "West":  ["4to1_0", "4to1_1"]
        }

        for episode in range(num_episodes):
            print(f"\n=== Starting episode {episode + 1} ===")

            sumo_cmd = [self.sumo_binary, "-c", self.config, "--start", "--no-warnings"]
            traci.start(sumo_cmd)

            step = 0
            cumulative_reward = 0
            print_interval = 2  # seconds
            last_print_time = 0  # last simulation time we printed at

            while step < max_steps:
                traci.simulationStep()
                sim_time = traci.simulation.getTime()

                # ✅ Print vehicle count every 2 seconds
                if sim_time - last_print_time >= print_interval:
                    print(f"\n[Time: {sim_time:.1f}s] Incoming vehicle count per direction:")

                    for direction, lanes in direction_lanes.items():
                        incoming_total = 0
                        for lane in lanes:
                            try:
                                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                                incoming_total += len(vehicle_ids)
                            except Exception as e:
                                print(f"  Lane {lane}: Error - {e}")
                        print(f"  {direction}: {incoming_total} vehicles")
                    last_print_time = sim_time

                    state = self.get_state()
                    action = self.agent.choose_action(state)

                    self.apply_action(action)
                    reward = self.calculate_reward()
                    cumulative_reward += reward

                    next_state = self.get_state()
                    self.agent.learn(state, action, reward, next_state)

                    lane_counts, waiting_times = self.count_waiting_vehicles()
                    total_wait_time = sum(waiting_times.values())
                    total_vehicles = sum(lane_counts.values())

                    log_entry = {
                        "Episode": episode + 1,
                        "Step": step,
                        "Action": action,
                        "Reward": reward,
                        "CumulativeReward": cumulative_reward,
                        "TotalWaitingTime": total_wait_time,
                        "AvgWaitingTimePerVeh": (total_wait_time / total_vehicles) if total_vehicles else 0
                    }

                    for i, (lane, count) in enumerate(lane_counts.items()):
                        log_entry[f"Lane_{i}_Count"] = count
                        log_entry[f"Lane_{i}_WaitTime"] = waiting_times[lane]

                    self.data_log.append(log_entry)

                    step += 1

            traci.close()
            print(f"Episode {episode + 1} finished.")

        # Save output to Excel
        df = pd.DataFrame(self.data_log)
        os.makedirs("data", exist_ok=True)
        df.to_excel(f"data/{self.agent.name}_results.xlsx", index=False)
        print(f"Results saved to data/{self.agent.name}_results.xlsx")

