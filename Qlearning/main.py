# main.py
from traffic_env import TrafficEnv
from q_agent import QLearningAgent
import numpy as np
import csv
import os
import time

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set GUI mode - change to False for faster training without GUI
USE_GUI = True

env = TrafficEnv("intersection.sumocfg", gui=USE_GUI)
agent = QLearningAgent(state_size=4, action_size=2)

EPISODES = 50
log_file = "logs/training_log.csv"

# Performance tracking
performance_history = []
best_performance = float('-inf')

print("="*80)
print("TRAFFIC LIGHT OPTIMIZATION USING Q-LEARNING")
print("="*80)
print(f"Episodes: {EPISODES}")
print(f"GUI Mode: {'ON' if USE_GUI else 'OFF'}")
print(f"State Size: {agent.state_size} (Queue lengths for 4 directions)")
print(f"Action Size: {agent.action_size} (Keep/Switch phase)")
print("-"*80)

try:
    with open(log_file, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total_Reward", "Avg_Queue_Length", "Max_Queue_Length", 
                        "Total_Waiting_Time", "Avg_Waiting_Time", "Phase_Switches", 
                        "Throughput", "Epsilon", "Q_Table_Size"])
        
        for ep in range(EPISODES):
            episode_start_time = time.time()
            
            try:
                print(f"\n--- EPISODE {ep + 1}/{EPISODES} ---")
                
                state = env.reset()
                done = False
                total_reward = 0
                step_count = 0
                phase_switches = 0
                queue_lengths = []
                waiting_times = []
                
                # Track initial state
                initial_vehicles = env.get_total_vehicles()
                print(f"Initial vehicles in simulation: {initial_vehicles}")
                
                while not done and step_count < 1000:
                    action = agent.choose_action(state)
                    
                    # Track phase switches
                    old_phase = env.current_phase
                    next_state, reward, done = env.step(action)
                    if env.current_phase != old_phase:
                        phase_switches += 1
                    
                    agent.learn(state, action, reward, next_state)
                    
                    # Collect performance metrics
                    queue_lengths.append(sum(next_state))
                    waiting_times.append(env.get_total_waiting_time())
                    
                    state = next_state
                    total_reward += reward
                    step_count += 1
                    
                    # Print progress every 100 steps
                    if step_count % 100 == 0:
                        current_queue = sum(state)
                        current_waiting = env.get_total_waiting_time()
                        current_vehicles = env.get_total_vehicles()
                        print(f"  Step {step_count}: Queue={current_queue}, Waiting={current_waiting:.1f}s, Vehicles={current_vehicles}, Reward={reward:.2f}")
                
                # Calculate episode metrics
                final_vehicles = env.get_total_vehicles()
                throughput = initial_vehicles - final_vehicles
                avg_queue = np.mean(queue_lengths) if queue_lengths else 0
                max_queue = max(queue_lengths) if queue_lengths else 0
                total_waiting = waiting_times[-1] if waiting_times else 0
                avg_waiting = np.mean(waiting_times) if waiting_times else 0
                
                episode_time = time.time() - episode_start_time
                
                # Performance summary
                print(f"\n📊 EPISODE {ep + 1} RESULTS:")
                print(f"  Duration: {episode_time:.1f}s")
                print(f"  Total Reward: {total_reward:.2f}")
                print(f"  Steps Taken: {step_count}")
                print(f"  Phase Switches: {phase_switches}")
                print(f"  Vehicle Throughput: {throughput}")
                print(f"  Average Queue Length: {avg_queue:.2f}")
                print(f"  Maximum Queue Length: {max_queue}")
                print(f"  Total Waiting Time: {total_waiting:.1f}s")
                print(f"  Average Waiting Time: {avg_waiting:.1f}s")
                print(f"  Exploration Rate (ε): {agent.epsilon:.3f}")
                print(f"  Q-Table Size: {len(agent.q_table)} states")
                
                # Performance comparison
                current_performance = total_reward
                if current_performance > best_performance:
                    best_performance = current_performance
                    print(f"  🎉 NEW BEST PERFORMANCE! (Previous: {best_performance:.2f})")
                
                performance_history.append({
                    'episode': ep + 1,
                    'reward': total_reward,
                    'avg_queue': avg_queue,
                    'throughput': throughput,
                    'waiting_time': total_waiting
                })
                
                # Write to CSV
                writer.writerow([ep + 1, total_reward, avg_queue, max_queue, 
                               total_waiting, avg_waiting, phase_switches, 
                               throughput, agent.epsilon, len(agent.q_table)])
                
                # Show improvement trend every 10 episodes
                if ep > 0 and (ep + 1) % 10 == 0:
                    recent_performance = np.mean([p['reward'] for p in performance_history[-10:]])
                    early_performance = np.mean([p['reward'] for p in performance_history[:10]])
                    improvement = recent_performance - early_performance
                    print(f"\n📈 IMPROVEMENT ANALYSIS (Last 10 vs First 10 episodes):")
                    print(f"  Reward Improvement: {improvement:.2f}")
                    print(f"  Recent Avg Queue: {np.mean([p['avg_queue'] for p in performance_history[-10:]]):.2f}")
                    print(f"  Early Avg Queue: {np.mean([p['avg_queue'] for p in performance_history[:10]]):.2f}")
                
            except Exception as e:
                print(f"❌ Error in episode {ep + 1}: {e}")
                continue
            finally:
                env.close()
                if USE_GUI:
                    time.sleep(1)  # Brief pause between episodes for GUI

except Exception as e:
    print(f"❌ Error creating log file: {e}")
finally:
    env.close()

# Final summary
print("\n" + "="*80)
print("🏁 TRAINING COMPLETED!")
print("="*80)
if performance_history:
    final_avg_reward = np.mean([p['reward'] for p in performance_history[-10:]])
    initial_avg_reward = np.mean([p['reward'] for p in performance_history[:10]])
    total_improvement = final_avg_reward - initial_avg_reward
    
    print(f"📊 FINAL PERFORMANCE SUMMARY:")
    print(f"  Episodes Completed: {len(performance_history)}")
    print(f"  Best Episode Reward: {best_performance:.2f}")
    print(f"  Initial Avg Reward (First 10): {initial_avg_reward:.2f}")
    print(f"  Final Avg Reward (Last 10): {final_avg_reward:.2f}")
    print(f"  Total Improvement: {total_improvement:.2f}")
    print(f"  Final Q-Table Size: {len(agent.q_table)} states learned")
    
    # Save the trained model
    agent.save_model("logs/trained_model.pkl")
    print(f"  Model saved to: logs/trained_model.pkl")

print(f"  Training log saved to: {log_file}")
print("="*80)
