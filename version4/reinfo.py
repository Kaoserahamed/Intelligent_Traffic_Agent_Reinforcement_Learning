import os
import sys
import traci
import random
import numpy as np

# Ensure SUMO tools are in the Python path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise ValueError("SUMO_HOME environment variable not set")

class TrafficLightQLearning:
    def __init__(self, num_states, num_actions):
        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        
        # Q-table initialization
        self.q_table = np.zeros((num_states, num_actions))
    
    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.q_table.shape[1] - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate, 
            self.exploration_rate * self.exploration_decay
        )

class SumoTrafficRL:
    def __init__(self, sumo_config):
        self.sumo_config = sumo_config
        self.traffic_light_id = "1"

        # Define phases
        self.phases = [
            "GGGgrrrrrrrrrrrr",  # North-South green
            "rrrrGGGgrrrrrrrr",  # East-West green
            "rrrrrrrrGGGgrrrr",  # Diagonal top-left to bottom-right green
            "rrrrrrrrrrrrGGGg"   # Diagonal bottom-left to top-right green
        ]

        # Yellow phases
        self.yellow_phases = [
            "yyyyrrrrrrrrrrrr",  
            "rrrryyyyrrrrrrrr",  
            "rrrrrrrryyyyrrrr",  
            "rrrrrrrrrrrryyyy"   
        ]

        # All red phase
        self.all_red_phase = "rrrrrrrrrrrrrrrr"

        # Phase parameters
        self.min_phase_duration = 42
        self.max_phase_duration = 48

        # Lanes for each phase
        self.phase_lanes = {
            0: ["1to3_0", "3to1_0"],  
            1: ["1to2_0", "2to1_0"],  
            2: ["1to4_0", "4to1_0"],  
            3: ["1to5_0", "5to1_0"]   
        }

        # RL Agent
        self.rl_agent = TrafficLightQLearning(
            num_states=10,
            num_actions=len(self.phases)
        )

        # Current phase tracking
        self.current_phase = None
        self.current_phase_duration = 0

    def count_waiting_vehicles(self, traci_instance):
        """
        Count the number of waiting vehicles in each phase.
        """
        waiting_counts = {phase: 0 for phase in self.phase_lanes}

        for phase, lanes in self.phase_lanes.items():
            waiting_count = 0
            for lane in lanes:
                try:
                    waiting_count += sum(
                        1 for vid in traci_instance.lane.getLastStepVehicleIDs(lane)
                        if traci_instance.vehicle.getSpeed(vid) < 0.1
                    )
                except traci.exceptions.TraCIException:
                    print(f"Warning: Could not count waiting vehicles in lane {lane}")

            waiting_counts[phase] = waiting_count

        print(f"Real-time waiting counts: {waiting_counts}")
        return waiting_counts

    def select_next_phase(self, traci_instance, current_phase):
        """
        Select next phase based on real-time waiting vehicles.
        """
        waiting_counts = self.count_waiting_vehicles(traci_instance)

        # Identify the phase with the highest number of waiting vehicles
        next_phase = max(waiting_counts, key=waiting_counts.get)

        # If the selected phase is the same as the current phase, check second highest
        if next_phase == current_phase:
            sorted_phases = sorted(waiting_counts.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_phases) > 1:
                next_phase = sorted_phases[1][0]  # Select the second most congested phase

        print(f"Selected next phase: {next_phase}")
        return next_phase

    def set_traffic_light_phase(self, traci_instance, action):
        """
        Set traffic light phase with transitions.
        """
        try:
            traci_instance.trafficlight.setRedYellowGreenState(
                self.traffic_light_id, self.yellow_phases[action]
            )
            traci_instance.simulationStep()

            traci_instance.trafficlight.setRedYellowGreenState(
                self.traffic_light_id, self.all_red_phase
            )
            traci_instance.simulationStep()

            traci_instance.trafficlight.setRedYellowGreenState(
                self.traffic_light_id, self.phases[action]
            )
        except traci.exceptions.TraCIException as e:
            print(f"Error setting traffic light phase: {e}")

    def run_simulation(self, num_episodes=100, max_steps=3600):
        sumo_binary = "sumo-gui"  

        for episode in range(num_episodes):
            self.current_phase = random.randint(0, len(self.phases) - 1)
            self.current_phase_duration = 0

            sumo_cmd = [
                sumo_binary,
                "-c", self.sumo_config,
                "--start",
                "--no-step-log", 
                "--no-warnings"
            ]

            try:
                traci.start(sumo_cmd)

                step = 0
                total_reward = 0

                while step < max_steps:
                    if traci.simulation.getMinExpectedNumber() <= 0:
                        print("No more vehicles in simulation")
                        break

                    traci.simulationStep()
                    self.current_phase_duration += 1

                    if self.current_phase_duration >= self.min_phase_duration:
                        waiting_counts = self.count_waiting_vehicles(traci)
                        current_state = min(sum(waiting_counts.values()) // 5, 9)

                        reward = -sum(waiting_counts.values())
                        total_reward += reward

                        next_phase = self.select_next_phase(traci, self.current_phase)

                        traci.simulationStep()
                        next_waiting_counts = self.count_waiting_vehicles(traci)
                        next_state = min(sum(next_waiting_counts.values()) // 5, 9)

                        self.rl_agent.update_q_table(
                            current_state, 
                            self.current_phase, 
                            reward, 
                            next_state
                        )

                        self.set_traffic_light_phase(traci, next_phase)
                        self.current_phase = next_phase
                        self.current_phase_duration = 0

                    step += 1

                print(f"Episode {episode + 1} completed. Total Reward: {total_reward}")
            
            except Exception as e:
                print(f"Simulation Error: {e}")
            finally:
                if traci.isLoaded():
                    traci.close()

def main():
    sumo_config = "intersection.sumocfg"

    try:
        traffic_controller = SumoTrafficRL(sumo_config)
        traffic_controller.run_simulation(num_episodes=50, max_steps=3600)
    
    except Exception as e:
        print(f"Error during simulation: {e}")

if __name__ == "__main__":
    main()
