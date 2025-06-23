from traffic_env import TrafficEnv
from q_agent import QLearningAgent
from dqn_agent import DQNAgent
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
    'EPISODES': 200,
    'USE_GUI': False,
    'MAX_STEPS_PER_EPISODE': 3600,
    'SAVE_INTERVAL': 25,
    'EVALUATION_INTERVAL': 50,
    'EARLY_STOPPING_PATIENCE': 50,  
}

# Agent selection setup
AGENTS = [
    {"name": "Q-Learning", "class": QLearningAgent, "model_path": "logs/best_q_model.pkl"},
    {"name": "DQN", "class": DQNAgent, "model_path": "logs/best_dqn_model.pth"},
]
SELECTED_AGENT_INDEX = 1  # Change to 1 to use DQNAgent


def create_training_log():
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
    return np.mean(values) if values else default


def plot_training_progress(performance_history, save_path="plots/training_progress.png"):
    if len(performance_history) < 5:
        return
    try:
        episodes = [p['episode'] for p in performance_history]
        rewards = [p['reward'] for p in performance_history]
        queues = [p['avg_queue'] for p in performance_history]
        waitings = [p['avg_waiting'] for p in performance_history]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        ax1.plot(episodes, rewards, 'b-', alpha=0.7, label='Episode Reward')
        if len(rewards) >= 10:
            window = min(10, len(rewards) // 2)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            moving_episodes = episodes[window-1:]
            ax1.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Average')
        ax1.set_title('Training Progress: Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True)

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

        ax3.plot(episodes, waitings, 'm-', alpha=0.7, label='Avg Waiting Time (s)')
        if len(waitings) >= 10:
            window = min(10, len(waitings) // 2)
            moving_avg = np.convolve(waitings, np.ones(window)/window, mode='valid')
            moving_episodes = episodes[window-1:]
            ax3.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Average')
        ax3.set_title('Training Progress: Waiting Time')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Waiting Time (s)')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating plot: {e}")


# rest of the code remains unchanged


def main():
    print("="*80)
    print("🚦 ENHANCED TRAFFIC LIGHT CONTROL TRAINING")
    print("="*80)

    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    print("-"*80)

    # Select agent class and model path
    selected = AGENTS[SELECTED_AGENT_INDEX]
    AgentClass = selected["class"]
    model_path = selected["model_path"]
    print(f"🤖 Selected Agent: {selected['name']}")

    env = TrafficEnv("intersection.sumocfg", 
                     max_steps=CONFIG['MAX_STEPS_PER_EPISODE'], 
                     gui=CONFIG['USE_GUI'])
    agent = AgentClass(state_size=8, action_size=2)

    if os.path.exists(model_path):
        agent.load_model(model_path)

    log_file = create_training_log()
    print(f"📝 Logging to: {log_file}")

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
                    state = env.reset()
                    if state is None:
                        print("❌ Failed to reset environment")
                        continue

                    total_reward = 0
                    step_count = 0
                    phase_switches = 0
                    metrics_history = []

                    for step in range(200):
                        action = agent.choose_action(state)
                        old_phase = env.current_phase
                        next_state, reward, done = env.step(action)
                        if env.current_phase != old_phase:
                            phase_switches += 1

                        agent.learn(state, action, reward, next_state, done)

                        metrics = env.get_traffic_metrics()
                        metrics_history.append(metrics)

                        state = next_state
                        total_reward += reward
                        step_count += 1

                        if step % 50 == 0:
                            print(f"   Step {step}: Reward={reward:.1f}, Queue={metrics['total_queue']}, "
                                  f"Speed={metrics['avg_speed']:.1f}, Vehicles={metrics['total_vehicles']}")

                        if done:
                            break

                    episode_duration = (time.time() - episode_start_time) / 60
                    avg_queue = safe_mean([m['total_queue'] for m in metrics_history])
                    max_queue = max([m['total_queue'] for m in metrics_history], default=0)
                    avg_waiting = safe_mean([m['avg_waiting_time'] for m in metrics_history])
                    avg_speed = safe_mean([m['avg_speed'] for m in metrics_history])
                    throughput = max([m['throughput'] for m in metrics_history], default=0)
                    agent_stats = agent.get_stats()

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

                    if total_reward > best_reward:
                        best_reward = total_reward
                        no_improvement_count = 0
                        agent.save_model(model_path)
                        print(f"   🎉 NEW BEST! Saved model.")
                    else:
                        no_improvement_count += 1

                    performance_data = {
                        'episode': episode + 1,
                        'reward': total_reward,
                        'avg_queue': avg_queue,
                        'avg_waiting': avg_waiting,
                        'throughput': throughput
                    }
                    performance_history.append(performance_data)

                    writer.writerow([
                        episode + 1, episode_duration, total_reward, avg_queue, max_queue,
                        avg_waiting, avg_speed, throughput, phase_switches,
                        agent.epsilon, agent_stats['q_table_size']
                    ])

                    if (episode + 1) % CONFIG['SAVE_INTERVAL'] == 0:
                        checkpoint_path = f"logs/checkpoint_ep{episode + 1}.pkl"
                        agent.save_model(checkpoint_path)
                        plot_training_progress(performance_history)
                        print(f"   💾 Checkpoint and plot saved")

                    if CONFIG['EARLY_STOPPING_PATIENCE'] is not None and no_improvement_count >= CONFIG['EARLY_STOPPING_PATIENCE']:
                        print(f"\n⏹️ Early stopping after {no_improvement_count} episodes without improvement")
                        break

                except Exception as e:
                    print(f"❌ Episode {episode + 1} error: {e}")
                    continue
                finally:
                    env.close()
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training error: {e}")
    finally:
        env.close()

    total_time = (time.time() - training_start_time) / 3600

    print("\n" + "="*80)
    print("🏁 TRAINING COMPLETED!")
    print("="*80)

    if performance_history:
        print(f"📊 FINAL STATISTICS:")
        print(f"   Episodes: {len(performance_history)}")
        print(f"   Training time: {total_time:.2f} hours")
        print(f"   Best reward: {best_reward:.2f}")
        print(f"   Final epsilon: {agent.epsilon:.4f}")
        print(f"   States learned: {agent.get_stats()['q_table_size']}")
        plot_training_progress(performance_history, "plots/final_training_progress.png")
        final_model_path = f"logs/final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        agent.save_model(final_model_path)
        print(f"\n💾 FILES SAVED:")
        print(f"   Best model: {model_path}")
        print(f"   Final model: {final_model_path}")
        print(f"   Training log: {log_file}")
        print(f"   Training plots: plots/")

if __name__ == "__main__":
    main()
