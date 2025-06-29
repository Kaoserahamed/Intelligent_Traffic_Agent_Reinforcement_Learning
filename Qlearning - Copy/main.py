from traffic_env import TrafficEnv
from q_agent import QLearningAgent
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from ppo_agent import PPOAgent
import numpy as np
import csv
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Agent configuration
AGENTS = [
    {"name": "Q-Learning", "class": QLearningAgent, "model_path": "logs/Q-Learning/best_q_model.pkl"},
    {"name": "DQN", "class": DQNAgent, "model_path": "logs/DQN/best_dqn_model.pth"},
    {"name": "DoubleDQN", "class": DoubleDQNAgent, "model_path": "logs/DoubleDQN/best_double_dqn_model.pth"},
    {"name": "PPO", "class": PPOAgent, "model_path": "logs/PPO/best_ppo_model.pth", "is_ppo": True}
]

# Training configuration
CONFIG = {
    'EPISODES': 200,
    'USE_GUI': False,
    'MAX_STEPS_PER_EPISODE': 5000,
    'SAVE_INTERVAL': 25,
    'EARLY_STOPPING_PATIENCE': 100,
}

def create_training_log(log_dir, agent_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/training_log_{agent_name}_{timestamp}.csv"
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

def plot_training_progress(performance_history, save_path):
    if len(performance_history) < 5:
        return
    try:
        episodes = [p['episode'] for p in performance_history]
        rewards = [p['reward'] for p in performance_history]
        queues = [p['avg_queue'] for p in performance_history]
        waitings = [p['avg_waiting'] for p in performance_history]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        def plot_metric(ax, values, label, color, ylabel):
            ax.plot(episodes, values, color+'-', alpha=0.7, label=label)
            if len(values) >= 10:
                window = min(10, len(values) // 2)
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                moving_episodes = episodes[window-1:]
                ax.plot(moving_episodes, moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Avg')
            ax.set_title(f'Training Progress: {label}')
            ax.set_xlabel('Episode')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)

        plot_metric(ax1, rewards, 'Episode Reward', 'b', 'Total Reward')
        plot_metric(ax2, queues, 'Avg Queue Length', 'g', 'Average Queue Length')
        plot_metric(ax3, waitings, 'Avg Waiting Time (s)', 'm', 'Average Waiting Time (s)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating plot: {e}")

def train_agent(selected):
    AGENT_NAME = selected["name"].replace(" ", "_")
    LOG_DIR = f"logs/{AGENT_NAME}"
    PLOT_DIR = f"plots/{AGENT_NAME}"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("="*80)
    print(f"🚦 TRAINING AGENT: {selected['name']}")
    print("="*80)

    env = TrafficEnv("intersection.sumocfg", max_steps=CONFIG['MAX_STEPS_PER_EPISODE'], gui=CONFIG['USE_GUI'])
    state = env.reset()
    if state is None:
        print("❌ Failed to initialize environment")
        return

    agent = selected["class"](state_size=len(state), action_size=4)

    if os.path.exists(selected["model_path"]):
        agent.load_model(selected["model_path"])

    log_file = create_training_log(LOG_DIR, AGENT_NAME)
    performance_history = []
    best_reward = float('-inf')
    no_improvement_count = 0

    with open(log_file, "a", newline='') as file:
        writer = csv.writer(file)

        for episode in range(CONFIG['EPISODES']):
            print(f"\n{'='*15} EPISODE {episode + 1}/{CONFIG['EPISODES']} {'='*15}")
            state = env.reset()
            if state is None:
                print("❌ Failed to reset environment")
                continue

            episode_reward = 0
            phase_switches = 0
            metrics_history = []

            done = False
            while not done:
                if selected.get("is_ppo"):
                    action, log_prob, value = agent.choose_action(state)
                else:
                    action = agent.choose_action(state)

                old_phase = env.current_phase
                next_state, reward, done = env.step(action)

                if env.current_phase != old_phase:
                    phase_switches += 1

                metrics = env.get_traffic_metrics()
                metrics_history.append(metrics)

                if selected.get("is_ppo"):
                    agent.store_transition(state, action, reward, next_state, done, log_prob, value)
                else:
                    agent.learn(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward

            if selected.get("is_ppo"):
                agent.update()

            # Stats and logging
            avg_queue = safe_mean([m['total_queue'] for m in metrics_history])
            max_queue = max([m['total_queue'] for m in metrics_history], default=0)
            avg_waiting = safe_mean([m['avg_waiting_time'] for m in metrics_history])
            avg_speed = safe_mean([m['avg_speed'] for m in metrics_history])
            throughput = max([m['throughput'] for m in metrics_history], default=0)
            episode_duration = CONFIG['MAX_STEPS_PER_EPISODE'] / 60

            print(f"   Reward: {episode_reward:.2f}, Avg Queue: {avg_queue:.2f}, Waiting: {avg_waiting:.2f}s")

            if episode_reward > best_reward:
                best_reward = episode_reward
                no_improvement_count = 0
                agent.save_model(selected["model_path"])
                print("   🎉 NEW BEST MODEL SAVED")
            else:
                no_improvement_count += 1

            stats = agent.get_stats() if not selected.get("is_ppo") else {'q_table_size': '-'}

            writer.writerow([
                episode + 1, episode_duration, episode_reward, avg_queue, max_queue,
                avg_waiting, avg_speed, throughput, phase_switches,
                getattr(agent, 'epsilon', '-'), stats['q_table_size']
            ])

            performance_history.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'avg_queue': avg_queue,
                'avg_waiting': avg_waiting,
                'throughput': throughput
            })

            if (episode + 1) % CONFIG['SAVE_INTERVAL'] == 0:
                plot_training_progress(performance_history, save_path=f"{PLOT_DIR}/training_progress.png")

            if CONFIG['EARLY_STOPPING_PATIENCE'] and no_improvement_count >= CONFIG['EARLY_STOPPING_PATIENCE']:
                print("⏹️ Early stopping: No improvement")
                break

            env.close()
            time.sleep(0.1)

    print(f"🏁 Training complete for {selected['name']}")
    plot_training_progress(performance_history, f"{PLOT_DIR}/final_training_progress.png")

def main():
    for agent_config in AGENTS:
        train_agent(agent_config)

if __name__ == "__main__":
    main()
