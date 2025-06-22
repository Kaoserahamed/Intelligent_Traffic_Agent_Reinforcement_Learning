from traffic_env import TrafficEnv
from q_agent import QLearningAgent
import numpy as np
import csv
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Training configuration
CONFIG = {
    'EPISODES': 200,  # Reduced for faster testing
    'USE_GUI': False,  # No GUI for automated training
    'MAX_STEPS_PER_EPISODE': 3600,  # 1 hour simulation
    'SAVE_INTERVAL': 25,
    'EVALUATION_INTERVAL': 50,
    'EARLY_STOPPING_PATIENCE': 50,
}

def create_training_log():
    """Create CSV log file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_log_{timestamp}.csv"
    
    with open(log_file, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Episode", "Duration_Min", "Total_Reward", "Avg_Queue", 
            "Max_Queue", "Avg_Waiting_Time", "Avg_Speed", "Throughput",
            "Phase_Switches", "Epsilon", "Q_Table_Size"
        ])
    
    return log_file

def safe_mean(values, default=0):
    """Safe mean calculation"""
    return np.mean(values) if values else default

def plot_training_progress(performance_history, save_path="plots/training_progress.png"):
    """Create training progress plot with error handling"""
    if len(performance_history) < 5:
        return
    
    try:
        episodes = [p['episode'] for p in performance_history]
        rewards = [p['reward'] for p in performance_history]
        queues = [p['avg_queue'] for p in performance_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Rewards plot
        ax1.plot(episodes, rewards, 'b-', alpha=0.7, label='Episode Reward')
        if len(rewards) >= 10:
            # Moving average
            window = min(10, len(rewards) // 2)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            moving_episodes = episodes[window-1:]
            ax1.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Average')
        
        ax1.set_title('Training Progress: Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True)
        
        # Queue lengths plot
        ax2.plot(episodes, queues, 'g-', alpha=0.7, label='Avg Queue Length')
        if len(queues) >= 10:
            window = min(10, len(queues) // 2)
            moving_avg = np.convolve(queues, np.ones(window)/window, mode='valid')
            moving_episodes = episodes[window-1:]
            ax2.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Average')
        
        ax2.set_title('Training Progress: Queue Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Queue Length')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating plot: {e}")

def main():
    print("="*80)
    print("🚦 ENHANCED TRAFFIC LIGHT Q-LEARNING")
    print("="*80)
    
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    print("-"*80)
    
    # Initialize environment and agent
    env = TrafficEnv("intersection.sumocfg", 
                     max_steps=CONFIG['MAX_STEPS_PER_EPISODE'], 
                     gui=CONFIG['USE_GUI'])
    agent = QLearningAgent(state_size=8, action_size=2)
    
    # Load existing model if available
    model_path = "logs/best_model.pkl"
    if os.path.exists(model_path):
        agent.load_model(model_path)
    
    # Create log file
    log_file = create_training_log()
    print(f"📝 Logging to: {log_file}")
    
    # Training tracking
    performance_history = []
    best_reward = float('-inf')
    no_improvement_count = 0
    training_start_time = time.time()
    
    try:
        with open(log_file, "a", newline='') as file:
            writer = csv.writer(file)
            
            for episode in range(CONFIG['EPISODES']):
                episode_start_time = time.time()
                
                print(f"\n{'='*15} EPISODE {episode + 1}/{CONFIG['EPISODES']} {'='*15}")
                
                try:
                    # Reset environment
                    state = env.reset()
                    if state is None:
                        print("❌ Failed to reset environment")
                        continue
                    
                    total_reward = 0
                    step_count = 0
                    phase_switches = 0
                    metrics_history = []
                    
                    # Episode loop
                    for step in range(200):  # Max 200 steps per episode
                        # Choose and execute action
                        action = agent.choose_action(state)
                        
                        # Track phase switches
                        old_phase = env.current_phase
                        next_state, reward, done = env.step(action)
                        if env.current_phase != old_phase:
                            phase_switches += 1
                        
                        # Learn
                        agent.learn(state, action, reward, next_state, done)
                        
                        # Update metrics
                        metrics = env.get_traffic_metrics()
                        metrics_history.append(metrics)
                        
                        state = next_state
                        total_reward += reward
                        step_count += 1
                        
                        # Progress update
                        if step % 50 == 0:
                            print(f"   Step {step}: Reward={reward:.1f}, Queue={metrics['total_queue']}, "
                                  f"Speed={metrics['avg_speed']:.1f}, Vehicles={metrics['total_vehicles']}")
                        
                        if done:
                            break
                    
                    # Calculate episode statistics
                    episode_duration = (time.time() - episode_start_time) / 60
                    
                    if metrics_history:
                        avg_queue = safe_mean([m['total_queue'] for m in metrics_history])
                        max_queue = max([m['total_queue'] for m in metrics_history], default=0)
                        avg_waiting = safe_mean([m['avg_waiting_time'] for m in metrics_history])
                        avg_speed = safe_mean([m['avg_speed'] for m in metrics_history])
                        throughput = max([m['throughput'] for m in metrics_history], default=0)
                    else:
                        avg_queue = max_queue = avg_waiting = avg_speed = throughput = 0
                    
                    agent_stats = agent.get_stats()
                    
                    # Episode summary
                    print(f"\n📊 EPISODE {episode + 1} RESULTS:")
                    print(f"   Duration: {episode_duration:.1f} min")
                    print(f"   Total Reward: {total_reward:.2f}")
                    print(f"   Steps: {step_count}")
                    print(f"   Avg Queue: {avg_queue:.2f}")
                    print(f"   Max Queue: {max_queue}")
                    print(f"   Avg Waiting: {avg_waiting:.1f}s")
                    print(f"   Avg Speed: {avg_speed:.2f} m/s")
                    print(f"   Throughput: {throughput}")
                    print(f"   Phase Switches: {phase_switches}")
                    print(f"   Epsilon: {agent.epsilon:.4f}")
                    print(f"   Q-States: {agent_stats['q_table_size']}")
                    
                    # Track best performance
                    if total_reward > best_reward:
                        best_reward = total_reward
                        no_improvement_count = 0
                        agent.save_model(model_path)
                        print(f"   🎉 NEW BEST! Saved model.")
                    else:
                        no_improvement_count += 1
                    
                    # Store performance
                    performance_data = {
                        'episode': episode + 1,
                        'reward': total_reward,
                        'avg_queue': avg_queue,
                        'avg_waiting': avg_waiting,
                        'throughput': throughput
                    }
                    performance_history.append(performance_data)
                    
                    # Log to CSV
                    writer.writerow([
                        episode + 1, episode_duration, total_reward, avg_queue, max_queue,
                        avg_waiting, avg_speed, throughput, phase_switches,
                        agent.epsilon, agent_stats['q_table_size']
                    ])
                    
                    # Periodic updates
                    if (episode + 1) % CONFIG['SAVE_INTERVAL'] == 0:
                        checkpoint_path = f"logs/checkpoint_ep{episode + 1}.pkl"
                        agent.save_model(checkpoint_path)
                        plot_training_progress(performance_history)
                        print(f"   💾 Checkpoint and plot saved")
                    
                    # Progress analysis
                    if (episode + 1) % 25 == 0 and len(performance_history) >= 25:
                        recent_avg = np.mean([p['reward'] for p in performance_history[-10:]])
                        early_avg = np.mean([p['reward'] for p in performance_history[:10]])
                        improvement = recent_avg - early_avg
                        
                        print(f"\n📈 PROGRESS ANALYSIS:")
                        print(f"   Episodes: {episode + 1}")
                        print(f"   Best reward: {best_reward:.2f}")
                        print(f"   Improvement: {improvement:.2f}")
                        print(f"   No improvement: {no_improvement_count}")
                    
                    # Early stopping
                    if no_improvement_count >= CONFIG['EARLY_STOPPING_PATIENCE']:
                        print(f"\n⏹️ Early stopping after {no_improvement_count} episodes without improvement")
                        break
                    
                except Exception as e:
                    print(f"❌ Episode {episode + 1} error: {e}")
                    continue
                finally:
                    env.close()
                    time.sleep(0.1)  # Small delay
    
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training error: {e}")
    finally:
        env.close()
    
    # Final summary
    total_time = (time.time() - training_start_time) / 3600
    
    print("\n" + "="*80)
    print("🏁 TRAINING COMPLETED!")
    print("="*80)
    
    if performance_history:
        print(f"📊 FINAL STATISTICS:")
        print(f"   Episodes: {len(performance_history)}")
        print(f"   Training time: {total_time:.2f} hours")
        print(f"   Best reward: {best_reward:.2f}")
        
        if len(performance_history) >= 20:
            early_rewards = [p['reward'] for p in performance_history[:10]]
            late_rewards = [p['reward'] for p in performance_history[-10:]]
            improvement = np.mean(late_rewards) - np.mean(early_rewards)
            print(f"   Total improvement: {improvement:.2f}")
        
        print(f"   Final epsilon: {agent.epsilon:.4f}")
        print(f"   States learned: {agent.get_stats()['q_table_size']}")
        
        # Final plots
        plot_training_progress(performance_history, "plots/final_training_progress.png")
        
        # Save final model
        final_model_path = f"logs/final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        agent.save_model(final_model_path)
        
        print(f"\n💾 FILES SAVED:")
        print(f"   Best model: {model_path}")
        print(f"   Final model: {final_model_path}")
        print(f"   Training log: {log_file}")
        print(f"   Training plots: plots/")
        
        # Performance assessment
        assess_final_performance(performance_history, best_reward)
    
    print("="*80)

def assess_final_performance(performance_history, best_reward):
    """Assess final performance and provide recommendations"""
    print(f"\n🔍 PERFORMANCE ASSESSMENT:")
    
    if len(performance_history) < 10:
        print("   ⚠️ Insufficient data for assessment")
        return
    
    # Calculate improvements
    early_episodes = performance_history[:len(performance_history)//4]  # First 25%
    late_episodes = performance_history[-len(performance_history)//4:]   # Last 25%
    
    early_reward = np.mean([ep['reward'] for ep in early_episodes])
    late_reward = np.mean([ep['reward'] for ep in late_episodes])
    reward_improvement = late_reward - early_reward
    
    early_queue = np.mean([ep['avg_queue'] for ep in early_episodes])
    late_queue = np.mean([ep['avg_queue'] for ep in late_episodes])
    queue_improvement = early_queue - late_queue  # Reduction is positive
    
    # Assessment criteria
    performance_score = 0
    recommendations = []
    
    print(f"   Best Episode Reward: {best_reward:.2f}")
    print(f"   Average Reward Improvement: {reward_improvement:.2f}")
    print(f"   Average Queue Reduction: {queue_improvement:.2f}")
    
    # Reward assessment
    if reward_improvement > 50:
        print("   ✅ Reward Learning: EXCELLENT")
        performance_score += 30
    elif reward_improvement > 20:
        print("   ⚠️ Reward Learning: GOOD")
        performance_score += 20
    elif reward_improvement > 0:
        print("   ⚠️ Reward Learning: FAIR")
        performance_score += 10
    else:
        print("   ❌ Reward Learning: POOR")
        recommendations.append("Increase training episodes or adjust reward function")
    
    # Queue improvement assessment
    if queue_improvement > 1.0:
        print("   ✅ Traffic Flow: EXCELLENT")
        performance_score += 30
    elif queue_improvement > 0.5:
        print("   ⚠️ Traffic Flow: GOOD")
        performance_score += 20
    elif queue_improvement > 0:
        print("   ⚠️ Traffic Flow: FAIR")
        performance_score += 10
    else:
        print("   ❌ Traffic Flow: POOR")
        recommendations.append("Check SUMO configuration and vehicle routes")
    
    # Best reward assessment
    if best_reward > -500:
        print("   ✅ Peak Performance: EXCELLENT")
        performance_score += 40
    elif best_reward > -1000:
        print("   ⚠️ Peak Performance: GOOD")
        performance_score += 30
    elif best_reward > -2000:
        print("   ⚠️ Peak Performance: FAIR")
        performance_score += 20
    else:
        print("   ❌ Peak Performance: POOR")
        recommendations.append("Review state representation and action space")
    
    # Overall assessment
    print(f"\n🎯 OVERALL PERFORMANCE SCORE: {performance_score}/100")
    
    if performance_score >= 80:
        print("   🎉 EXCELLENT - Ready for real-world testing!")
        print("   💡 Consider fine-tuning hyperparameters for optimization")
    elif performance_score >= 60:
        print("   ✅ GOOD - Suitable for deployment with monitoring")
        print("   💡 Continue training or adjust parameters for better performance")
    elif performance_score >= 40:
        print("   ⚠️ FAIR - Needs improvement before deployment")
        print("   💡 Significant adjustments required")
    else:
        print("   ❌ POOR - Major issues need addressing")
        print("   💡 Review entire approach and configuration")
    
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Deployment readiness
    print(f"\n🚀 DEPLOYMENT READINESS:")
    if performance_score >= 70 and best_reward > -1000:
        print("   ✅ Model shows promising results for real-world application")
        print("   ✅ Consider A/B testing against current traffic control")
        print("   ⚠️ Monitor performance closely during initial deployment")
    else:
        print("   ❌ Model needs further development before deployment")
        print("   💡 Continue training with adjusted parameters")

if __name__ == "__main__":
    main()
        