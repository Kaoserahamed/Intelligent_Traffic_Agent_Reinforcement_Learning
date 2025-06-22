# interactive_test.py
from traffic_env import TrafficEnv
from q_agent import QLearningAgent
import time

def interactive_test():
    """Interactive testing where you can observe the model in action"""
    
    print("="*80)
    print("INTERACTIVE MODEL TESTING")
    print("="*80)
    print("This will run the trained model with GUI so you can observe its behavior.")
    print("Press Ctrl+C to stop at any time.")
    
    # Initialize environment with GUI
    env = TrafficEnv("intersection.sumocfg", gui=True)
    agent = QLearningAgent(state_size=4, action_size=2)
    
    # Load trained model
    try:
        agent.load_model("logs/trained_model.pkl")
        agent.epsilon = 0.0  # Pure exploitation
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    try:
        episode = 1
        while True:
            print(f"\n🚦 Starting Episode {episode}")
            print("Watch the simulation window to see the trained agent in action!")
            
            state = env.reset()
            done = False
            step_count = 0
            total_reward = 0
            
            while not done and step_count < 1000:
                # Get action from trained model
                action = agent.choose_action(state)
                
                # Execute action
                next_state, reward, done = env.step(action)
                
                # Print current status
                if step_count % 50 == 0:
                    action_name = "SWITCH" if action == 1 else "KEEP"
                    current_phase = env.current_phase
                    queue_total = sum(next_state)
                    waiting_time = env.get_total_waiting_time()
                    
                    print(f"  Step {step_count}: Action={action_name}, Phase={current_phase}, "
                          f"Queue={queue_total}, Waiting={waiting_time:.1f}s, Reward={reward:.2f}")
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                # Add small delay to make it easier to observe
                time.sleep(0.1)
            
            print(f"✅ Episode {episode} completed!")
            print(f"   Total Reward: {total_reward:.2f}")
            print(f"   Steps: {step_count}")
            
            # Ask user if they want to continue
            user_input = input("\nPress Enter to run another episode, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
            
            episode += 1
            env.close()
    
    except KeyboardInterrupt:
        print("\n🛑 Testing stopped by user")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    finally:
        env.close()
        print("Testing session ended.")

if __name__ == "__main__":
    interactive_test()