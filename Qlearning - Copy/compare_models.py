# compare_models.py
from traffic_env import TrafficEnv
from q_agent import QLearningAgent
import numpy as np
import random
import time

def test_random_policy(env, episodes=5):
    """Test random action policy as baseline"""
    results = []
    
    print("Testing Random Policy (Baseline)...")
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        queue_lengths = []
        
        while not done and step_count < 1000:
            action = random.randint(0, 1)  # Random action
            next_state, reward, done = env.step(action)
            
            queue_lengths.append(sum(next_state))
            state = next_state
            total_reward += reward
            step_count += 1
        
        avg_queue = np.mean(queue_lengths) if queue_lengths else 0
        results.append({
            'reward': total_reward,
            'avg_queue': avg_queue,
            'waiting_time': env.get_total_waiting_time()
        })
        
        env.close()
    
    return results

def test_fixed_timing_policy(env, episodes=5):
    """Test fixed timing policy (switch every 30 steps)"""
    results = []
    
    print("Testing Fixed Timing Policy...")
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        queue_lengths = []
        
        while not done and step_count < 1000:
            # Switch phase every 30 steps
            action = 1 if step_count % 30 == 0 else 0
            next_state, reward, done = env.step(action)
            
            queue_lengths.append(sum(next_state))
            state = next_state
            total_reward += reward
            step_count += 1
        
        avg_queue = np.mean(queue_lengths) if queue_lengths else 0
        results.append({
            'reward': total_reward,
            'avg_queue': avg_queue,
            'waiting_time': env.get_total_waiting_time()
        })
        
        env.close()
    
    return results

def compare_policies():
    """Compare trained model with baseline policies"""
    
    print("="*80)
    print("POLICY COMPARISON TEST")
    print("="*80)
    
    env = TrafficEnv("intersection.sumocfg", gui=False)  # No GUI for faster testing
    
    # Test 1: Random Policy
    random_results = test_random_policy(env, episodes=3)
    
    # Test 2: Fixed Timing Policy
    fixed_results = test_fixed_timing_policy(env, episodes=3)
    
    # Test 3: Trained Q-Learning Model
    agent = QLearningAgent(state_size=4, action_size=2)
    try:
        agent.load_model("logs/trained_model.pkl")
        agent.epsilon = 0.0  # No exploration during testing
        
        print("Testing Trained Q-Learning Model...")
        ql_results = []
        
        for ep in range(3):
            state = env.reset()
            done = False
            total_reward = 0
            step_count = 0
            queue_lengths = []
            
            while not done and step_count < 1000:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                
                queue_lengths.append(sum(next_state))
                state = next_state
                total_reward += reward
                step_count += 1
            
            avg_queue = np.mean(queue_lengths) if queue_lengths else 0
            ql_results.append({
                'reward': total_reward,
                'avg_queue': avg_queue,
                                'waiting_time': env.get_total_waiting_time()
            })
            
            env.close()
        
        # Compare Results
        print("\n" + "="*80)
        print("📊 COMPARISON RESULTS")
        print("="*80)
        
        policies = {
            'Random Policy': random_results,
            'Fixed Timing': fixed_results,
            'Q-Learning (Trained)': ql_results
        }
        
        for policy_name, results in policies.items():
            avg_reward = np.mean([r['reward'] for r in results])
            avg_queue = np.mean([r['avg_queue'] for r in results])
            avg_waiting = np.mean([r['waiting_time'] for r in results])
            
            print(f"\n{policy_name}:")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Queue Length: {avg_queue:.2f}")
            print(f"  Average Waiting Time: {avg_waiting:.1f}s")
        
        # Calculate improvements
        random_reward = np.mean([r['reward'] for r in random_results])
        ql_reward = np.mean([r['reward'] for r in ql_results])
        improvement = ((ql_reward - random_reward) / abs(random_reward)) * 100
        
        print(f"\n🎯 Q-Learning vs Random Policy:")
        print(f"  Reward Improvement: {improvement:.1f}%")
        
        random_queue = np.mean([r['avg_queue'] for r in random_results])
        ql_queue = np.mean([r['avg_queue'] for r in ql_results])
        queue_improvement = ((random_queue - ql_queue) / random_queue) * 100
        
        print(f"  Queue Length Reduction: {queue_improvement:.1f}%")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
    
    finally:
        env.close()

if __name__ == "__main__":
    compare_policies()