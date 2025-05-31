# main.py

from simulation_controller import SumoSimulation
from agents.dqn_agent import DQNTrafficAgent
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from utils import plot_performance

def run_with_agent(agent, label):
    sim = SumoSimulation(
        config="intersection.sumocfg",
        agent=agent,
        reward_method="default",
        sumo_binary=r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
    )
    sim.run(num_episodes=1, max_steps=100, output_excel=f"{label}_results.xlsx")
    plot_performance(f"data/{label}_results.xlsx")

def main():
    agents_to_test = {
        "DQN": DQNTrafficAgent(1, 4),
        "Q_Learning": QLearningAgent(10, 4),
        "Random": RandomAgent(4),
    }

    for label, agent in agents_to_test.items():
        print(f"\n=== Running simulation with {label} Agent ===")
        run_with_agent(agent, label)

if __name__ == "__main__":
    main()
