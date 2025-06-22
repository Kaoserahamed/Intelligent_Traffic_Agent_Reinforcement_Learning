import traci
import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
import json

# Import SUMO library
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class TrafficLightQLearning:
    """Q-Learning agent for traffic light control"""
    
    def __init__(self, intersection_id='1', learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.intersection_id = intersection_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Traffic light phases (actions)
        self.actions = {
            0: 'EW_GREEN',    # East-West green, North-South red
            1: 'NS_GREEN',    # North-South green, East-West red
            2: 'ALL_RED'      # All red (clearing phase)
        }
        
        # Q-table: state -> action -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_waiting_times = []
        self.episode_queue_lengths = []
        self.detailed_rewards = []
        
        # State discretization parameters
        self.max_vehicles_per_edge = 20
        self.waiting_time_threshold = 10
        
        # Incoming edges for monitoring
        self.incoming_edges = {
            'East': '2to1',
            'North': '3to1', 
            'West': '4to1',
            'South': '5to1'
        }
        
        # Phase timing
        self.min_phase_duration = 10  # Minimum green time
        self.max_phase_duration = 60  # Maximum green time
        self.yellow_duration = 3
        self.all_red_duration = 2
        
        self.current_phase = 0
        self.phase_time = 0
        self.last_action_time = 0
        
    def get_state(self):
        """Get current traffic state as a discrete state representation"""
        try:
            # Get vehicle counts on each incoming edge
            vehicle_counts = {}
            total_waiting_time = 0
            total_vehicles = 0
            
            for direction, edge_id in self.incoming_edges.items():
                # Vehicle count
                count = traci.edge.getLastStepVehicleNumber(edge_id)
                vehicle_counts[direction] = min(count, self.max_vehicles_per_edge)
                total_vehicles += count
                
                # Waiting time for vehicles on this edge
                vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
                for vehicle in vehicles:
                    waiting_time = traci.vehicle.getAccumulatedWaitingTime(vehicle)
                    total_waiting_time += waiting_time
            
            # Average waiting time
            avg_waiting_time = total_waiting_time / max(total_vehicles, 1)
            waiting_category = min(int(avg_waiting_time / self.waiting_time_threshold), 5)
            
            # Current traffic light phase
            current_phase = self.current_phase
            
            # Time since last phase change
            time_in_phase = min(self.phase_time // 10, 6)  # Discretize to 10-second intervals
            
            # Create state tuple
            state = (
                vehicle_counts['East'],
                vehicle_counts['North'], 
                vehicle_counts['West'],
                vehicle_counts['South'],
                waiting_category,
                current_phase,
                time_in_phase
            )
            
            return state
            
        except Exception as e:
            print(f"Error getting state: {e}")
            return (0, 0, 0, 0, 0, 0, 0)
    
    def calculate_reward(self):
        """Calculate reward based on traffic conditions"""
        try:
            total_waiting_time = 0
            total_vehicles = 0
            total_queue_length = 0
            stopped_vehicles = 0
            
            # Collect traffic metrics
            for direction, edge_id in self.incoming_edges.items():
                # Vehicle count and queue length
                count = traci.edge.getLastStepVehicleNumber(edge_id)
                total_vehicles += count
                total_queue_length += count
                
                # Get vehicles on this edge
                vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
                
                for vehicle in vehicles:
                    # Waiting time
                    waiting_time = traci.vehicle.getAccumulatedWaitingTime(vehicle)
                    total_waiting_time += waiting_time
                    
                    # Check if vehicle is stopped
                    speed = traci.vehicle.getSpeed(vehicle)
                    if speed < 0.5:  # Consider as stopped if speed < 0.5 m/s
                        stopped_vehicles += 1
            
            # Calculate individual reward components
            waiting_penalty = -total_waiting_time * 0.1
            queue_penalty = -total_queue_length * 2.0
            stopped_penalty = -stopped_vehicles * 1.5
            throughput_reward = max(0, (total_vehicles - stopped_vehicles)) * 1.0
            
            # Efficiency bonus (low waiting time with high throughput)
            if total_vehicles > 0:
                avg_waiting = total_waiting_time / total_vehicles
                efficiency_bonus = max(0, (20 - avg_waiting)) * 0.5
            else:
                efficiency_bonus = 0
            
            # Total reward
            total_reward = (waiting_penalty + queue_penalty + stopped_penalty + 
                          throughput_reward + efficiency_bonus)
            
            # Store detailed reward information
            reward_details = {
                'total_reward': total_reward,
                'waiting_penalty': waiting_penalty,
                'queue_penalty': queue_penalty,
                'stopped_penalty': stopped_penalty,
                'throughput_reward': throughput_reward,
                'efficiency_bonus': efficiency_bonus,
                'total_waiting_time': total_waiting_time,
                'total_vehicles': total_vehicles,
                'stopped_vehicles': stopped_vehicles,
                'avg_waiting_time': total_waiting_time / max(total_vehicles, 1)
            }
            
            return total_reward, reward_details
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0, {}
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(list(self.actions.keys()))
        else:
            # Exploitation: best known action
            q_values = [self.q_table[state][action] for action in self.actions.keys()]
            return np.argmax(q_values)
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning algorithm"""
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Maximum Q-value for next state
        next_q_values = [self.q_table[next_state][a] for a in self.actions.keys()]
        max_next_q = max(next_q_values) if next_q_values else 0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def set_traffic_light_phase(self, action):
        """Set traffic light phase based on action"""
        try:
            if action == 0:  # East-West Green
                traci.trafficlight.setPhase(self.intersection_id, 0)
            elif action == 1:  # North-South Green  
                traci.trafficlight.setPhase(self.intersection_id, 2)
            elif action == 2:  # All Red
                traci.trafficlight.setPhase(self.intersection_id, 1)
                
            self.current_phase = action
            
        except Exception as e:
            print(f"Error setting traffic light phase: {e}")
    
    def should_change_phase(self, current_time):
        """Determine if phase should be changed based on minimum/maximum durations"""
        time_in_current_phase = current_time - self.last_action_time
        
        # Minimum time constraint
        if time_in_current_phase < self.min_phase_duration:
            return False
            
        # Maximum time constraint
        if time_in_current_phase >= self.max_phase_duration:
            return True
            
        return True  # Allow phase change
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filename='traffic_q_model.pkl'):
        """Save Q-table and parameters"""
        model_data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_waiting_times': self.episode_waiting_times,
            'episode_queue_lengths': self.episode_queue_lengths
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='traffic_q_model.pkl'):
        """Load Q-table and parameters"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.episode_rewards = model_data.get('episode_rewards', [])
            self.episode_waiting_times = model_data.get('episode_waiting_times', [])
            self.episode_queue_lengths = model_data.get('episode_queue_lengths', [])
            
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"No existing model found at {filename}. Starting fresh.")


def run_q_learning_simulation(episodes=100, steps_per_episode=3600, 
                            save_model=True, load_existing=True):
    """Run Q-learning traffic control simulation"""
    
    # Initialize Q-learning agent
    agent = TrafficLightQLearning()
    
    # Load existing model if available
    if load_existing:
        agent.load_model()
    
    print("Starting Q-Learning Traffic Control Simulation")
    print("=" * 60)
    
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        print("-" * 40)
        
        # Start SUMO simulation
        traci.start(["sumo-gui", "-c", "intersection.sumocfg", "--start"])
        
        episode_reward = 0
        episode_waiting_time = 0
        episode_queue_length = 0
        step_count = 0
        
        # Get initial state
        state = agent.get_state()
        
        try:
            while step_count < steps_per_episode:
                # Choose action
                if agent.should_change_phase(step_count):
                    action = agent.choose_action(state)
                    agent.set_traffic_light_phase(action)
                    agent.last_action_time = step_count
                
                # Advance simulation
                traci.simulationStep()
                step_count += 1
                agent.phase_time += 1
                
                # Get next state and reward
                next_state = agent.get_state()
                reward, reward_details = agent.calculate_reward()
                
                # Update Q-table
                if agent.should_change_phase(step_count - 1):
                    agent.update_q_table(state, action, reward, next_state)
                
                # Track performance
                episode_reward += reward
                episode_waiting_time += reward_details.get('total_waiting_time', 0)
                episode_queue_length += reward_details.get('total_vehicles', 0)
                
                # Store detailed reward info periodically
                if step_count % 300 == 0:  # Every 5 minutes
                    agent.detailed_rewards.append({
                        'episode': episode + 1,
                        'step': step_count,
                        **reward_details
                    })
                
                # Display progress every 10 minutes
                if step_count % 600 == 0:
                    print(f"  Step {step_count}: Reward={reward:.2f}, "
                          f"Avg Waiting={reward_details.get('avg_waiting_time', 0):.2f}s, "
                          f"Vehicles={reward_details.get('total_vehicles', 0)}")
                
                state = next_state
                
        except Exception as e:
            print(f"Error in episode {episode + 1}: {e}")
        
        finally:
            traci.close()
        
        # Store episode metrics
        agent.episode_rewards.append(episode_reward)
        agent.episode_waiting_times.append(episode_waiting_time / steps_per_episode)
        agent.episode_queue_lengths.append(episode_queue_length / steps_per_episode)
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Print episode summary
        print(f"  Episode Summary:")
        print(f"    Total Reward: {episode_reward:.2f}")
        print(f"    Avg Waiting Time: {episode_waiting_time/steps_per_episode:.2f}s")
        print(f"    Avg Queue Length: {episode_queue_length/steps_per_episode:.2f}")
        print(f"    Epsilon: {agent.epsilon:.4f}")
        
        # Save model periodically
        if save_model and (episode + 1) % 10 == 0:
            agent.save_model(f'traffic_q_model_episode_{episode + 1}.pkl')
    
    # Final model save
    if save_model:
        agent.save_model('final_traffic_q_model.pkl')
    
    # Generate performance report
    generate_performance_report(agent)
    
    return agent


def generate_performance_report(agent):
    """Generate detailed performance analysis and visualizations"""
    
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS REPORT")
    print("=" * 60)
    
    if not agent.episode_rewards:
        print("No data available for analysis.")
        return
    
    # Performance metrics
    avg_reward = np.mean(agent.episode_rewards[-10:])  # Last 10 episodes
    avg_waiting = np.mean(agent.episode_waiting_times[-10:])
    avg_queue = np.mean(agent.episode_queue_lengths[-10:])
    
    print(f"\nFinal Performance Metrics (Last 10 Episodes):")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Waiting Time: {avg_waiting:.2f} seconds")
    print(f"  Average Queue Length: {avg_queue:.2f} vehicles")
    
    # Learning progress
    if len(agent.episode_rewards) > 1:
        improvement = agent.episode_rewards[-1] - agent.episode_rewards[0]
        print(f"  Total Improvement: {improvement:.2f}")
    
    # Detailed reward breakdown
    if agent.detailed_rewards:
        print(f"\nDetailed Reward Analysis:")
        recent_rewards = agent.detailed_rewards[-5:]  # Last 5 measurements
        
        for reward_info in recent_rewards:
            print(f"  Episode {reward_info['episode']}, Step {reward_info['step']}:")
            print(f"    Total Reward: {reward_info.get('total_reward', 0):.2f}")
            print(f"    Waiting Penalty: {reward_info.get('waiting_penalty', 0):.2f}")
            print(f"    Queue Penalty: {reward_info.get('queue_penalty', 0):.2f}")
            print(f"    Throughput Reward: {reward_info.get('throughput_reward', 0):.2f}")
            print(f"    Efficiency Bonus: {reward_info.get('efficiency_bonus', 0):.2f}")
            print(f"    Vehicles: {reward_info.get('total_vehicles', 0)}")
            print(f"    Avg Waiting: {reward_info.get('avg_waiting_time', 0):.2f}s")
            print()
    
    # Q-table statistics
    total_states = len(agent.q_table)
    total_q_values = sum(len(actions) for actions in agent.q_table.values())
    print(f"Q-Table Statistics:")
    print(f"  States Explored: {total_states}")
    print(f"  Total Q-Values: {total_q_values}")
    
    # Save detailed results
    results = {
        'episode_rewards': agent.episode_rewards,
        'episode_waiting_times': agent.episode_waiting_times,
        'episode_queue_lengths': agent.episode_queue_lengths,
        'detailed_rewards': agent.detailed_rewards,
        'final_epsilon': agent.epsilon,
        'q_table_size': total_states
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to 'training_results.json'")
    print("=" * 60)


def run_test_simulation(model_file='final_traffic_q_model.pkl', duration=3600):
    """Run a test simulation with trained model (no learning)"""
    
    print("Running Test Simulation with Trained Model")
    print("=" * 50)
    
    # Load trained agent
    agent = TrafficLightQLearning()
    agent.load_model(model_file)
    agent.epsilon = 0  # No exploration in test mode
    
    # Start SUMO
    traci.start(["sumo-gui", "-c", "intersection.sumocfg", "--start"])
    
    step = 0
    total_reward = 0
    
    try:
        while step < duration:
            # Get current state
            state = agent.get_state()
            
            # Choose best action (no exploration)
            action = agent.choose_action(state)
            
            # Apply action
            if agent.should_change_phase(step):
                agent.set_traffic_light_phase(action)
                agent.last_action_time = step
            
            # Advance simulation
            traci.simulationStep()
            
            # Calculate reward for monitoring
            reward, reward_details = agent.calculate_reward()
            total_reward += reward
            
            # Display status every 10 minutes
            if step % 600 == 0:
                print(f"Step {step}: Reward={reward:.2f}, "
                      f"Vehicles={reward_details.get('total_vehicles', 0)}, "
                      f"Avg Waiting={reward_details.get('avg_waiting_time', 0):.2f}s")
            
            step += 1
            agent.phase_time += 1
    
    finally:
        traci.close()
    
    print(f"\nTest Simulation Complete")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average Reward per Step: {total_reward/duration:.4f}")


if __name__ == "__main__":
    # Training mode
    print("Choose mode:")
    print("1. Train new model")
    print("2. Continue training existing model") 
    print("3. Test trained model")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        agent = run_q_learning_simulation(episodes=50, load_existing=False)
    elif choice == '2':
        agent = run_q_learning_simulation(episodes=20, load_existing=True)
    elif choice == '3':
        run_test_simulation()
    else:
        print("Invalid choice. Running default training...")
        agent = run_q_learning_simulation(episodes=30)