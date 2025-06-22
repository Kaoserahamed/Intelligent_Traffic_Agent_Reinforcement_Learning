# test_model.py
from traffic_env import TrafficEnv
from q_agent import QLearningAgent
import numpy as np
import csv
import os
import time

def test_trained_model(model_path="logs/trained_model.pkl", test_episodes=10, gui=True):
    """Test the trained Q-learning model"""
    
    print("="*80)
    print("TESTING TRAINED TRAFFIC LIGHT MODEL")
    print("="*80)
    
    # Initialize environment and agent
    env = TrafficEnv("intersection.sumocfg", gui=gui)
    agent = QLearningAgent(state_size=4, action_size=2)
    
    # Load the trained model
    try:
        agent.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"Q-Table size: {len(agent.q_table)} states")
        print(f"Final epsilon: {agent.epsilon:.3f}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Set epsilon to 0 for pure exploitation (no exploration during testing)
    agent.epsilon = 0.0
    
    # Test results storage
    test_results = []
    
    print(f"\nRunning {test_episodes} test episodes...")
    print("-"*80)
    
    for ep in range(test_episodes):
        print(f"\n🧪 TEST EPISODE {ep + 1}/{test_episodes}")
        
        try:
            state = env.reset()
            done = False
            total_reward = 0
            step_count = 0
            phase_switches = 0
            queue_lengths = []
            waiting_times = []
            
            # Track initial state
            initial_vehicles = env.get_total_vehicles()
            
            while not done and step_count < 1000:
                # Agent chooses action (pure exploitation, no exploration)
                action = agent.choose_action(state)
                
                # Track phase switches
                old_phase = env.current_phase
                next_state, reward, done = env.step(action)
                if env.current_phase != old_phase:
                    phase_switches += 1
                
                # Collect metrics
                queue_lengths.append(sum(next_state))
                waiting_times.append(env.get_total_waiting_time())
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                # Print progress
                if step_count % 200 == 0:
                    current_queue = sum(state)
                    current_waiting = env.get_total_waiting_time()
                    current_vehicles = env.get_total_vehicles()
                    print(f"  Step {step_count}: Queue={current_queue}, Waiting={current_waiting:.1f}s, Vehicles={current_vehicles}")
            
            # Calculate episode results
            final_vehicles = env.get_total_vehicles()
            throughput = initial_vehicles - final_vehicles
            avg_queue = np.mean(queue_lengths) if queue_lengths else 0
            max_queue = max(queue_lengths) if queue_lengths else 0
            total_waiting = waiting_times[-1] if waiting_times else 0
            avg_waiting = np.mean(waiting_times) if waiting_times else 0
            
            # Store results
            episode_result = {
                'episode': ep + 1,
                'total_reward': total_reward,
                'throughput': throughput,
                'avg_queue': avg_queue,
                'max_queue': max_queue,
                'total_waiting': total_waiting,
                'avg_waiting': avg_waiting,
                'phase_switches': phase_switches,
                'steps': step_count
            }
            test_results.append(episode_result)
            
            # Print episode summary
            print(f"  ✅ Episode {ep + 1} Results:")
            print(f"     Total Reward: {total_reward:.2f}")
            print(f"     Vehicle Throughput: {throughput}")
            print(f"     Avg Queue Length: {avg_queue:.2f}")
            print(f"     Total Waiting Time: {total_waiting:.1f}s")
            print(f"     Phase Switches: {phase_switches}")
            
        except Exception as e:
            print(f"❌ Error in test episode {ep + 1}: {e}")
        finally:
            env.close()
            if gui:
                time.sleep(1)
    
    # Calculate overall performance
    if test_results:
        print("\n" + "="*80)
        print("📊 TESTING RESULTS SUMMARY")
        print("="*80)
        
        avg_reward = np.mean([r['total_reward'] for r in test_results])
        avg_throughput = np.mean([r['throughput'] for r in test_results])
        avg_queue_length = np.mean([r['avg_queue'] for r in test_results])
        avg_waiting_time = np.mean([r['total_waiting'] for r in test_results])
        avg_phase_switches = np.mean([r['phase_switches'] for r in test_results])
        
        print(f"Average Performance over {len(test_results)} episodes:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Throughput: {avg_throughput:.1f} vehicles")
        print(f"  Average Queue Length: {avg_queue_length:.2f}")
        print(f"  Average Waiting Time: {avg_waiting_time:.1f}s")
        print(f"  Average Phase Switches: {avg_phase_switches:.1f}")
        
        # Save test results
        test_log_file = "logs/test_results.csv"
        with open(test_log_file, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Total_Reward", "Throughput", "Avg_Queue_Length", 
                           "Max_Queue_Length", "Total_Waiting_Time", "Avg_Waiting_Time", 
                           "Phase_Switches", "Steps"])
            
            for result in test_results:
                writer.writerow([result['episode'], result['total_reward'], 
                               result['throughput'], result['avg_queue'], result['max_queue'],
                               result['total_waiting'], result['avg_waiting'], 
                               result['phase_switches'], result['steps']])
        
        print(f"  Test results saved to: {test_log_file}")
        print("="*80)
        
        return test_results
    
    else:
        print("❌ No successful test episodes completed")
        return None

if __name__ == "__main__":
    # Test the model
    results = test_trained_model(
        model_path="logs/trained_model.pkl",
        test_episodes=5,  # Adjust as needed
        gui=True  # Set to False for faster testing
    )