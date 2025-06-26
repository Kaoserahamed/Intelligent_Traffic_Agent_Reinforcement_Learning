from traffic_env import TrafficEnv
from q_agent import QLearningAgent
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from fixed_agent import FixedTimeAgent
from random import choice
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time

# Configuration
AGENTS = [
    {"name": "Q-Learning", "class": QLearningAgent, "model_path": "logs/Q-Learning/best_q_model.pkl"},
    {"name": "DQN", "class": DQNAgent, "model_path": "logs/DQN/best_dqn_model.pth"},
    {"name": "DoubleDQN", "class": DoubleDQNAgent, "model_path": "logs/DoubleDQN/best_double_dqn_model.pth"},
]
FIXED_INTERVAL = 40
MAX_STEPS = 5000
RANDOM_SEED = 42
VEHICLES_PER_STEP = 1

INCOMING_EDGES = ['2to1', '3to1', '5to1', '4to1']
OUTGOING_EDGES = ['1to2', '1to3', '1to5', '1to4']
VEHICLE_TYPES = ['car', 'bus', 'emergency', 'truck', 'bike']

# 🔧 Generate vehicle schedule for every step
def generate_vehicle_schedule(steps=MAX_STEPS, vehicles_per_step=VEHICLES_PER_STEP, seed_value=RANDOM_SEED):
    random.seed(seed_value)
    schedule = []
    veh_count = 0
    for _ in range(steps):
        step_vehicles = []
        for _ in range(vehicles_per_step):
            from_edge = choice(INCOMING_EDGES)
            from_node = from_edge.split('to')[0]
            valid_to_edges = [e for e in OUTGOING_EDGES if e.split('to')[1] != from_node]
            to_edge = choice(valid_to_edges)
            veh_type = choice(VEHICLE_TYPES)
            step_vehicles.append({
                "veh_id": f"veh_{veh_count}",
                "from": from_edge,
                "to": to_edge,
                "type": veh_type
            })
            veh_count += 1
        schedule.append(step_vehicles)
    return schedule

# 🚗 Spawn vehicles for a step
def spawn_vehicle_plan(vehicle_plan):
    import traci
    for v in vehicle_plan:
        route_id = f"{v['from']}_to_{v['to']}"
        if route_id not in traci.route.getIDList():
            traci.route.add(route_id, [v["from"], v["to"]])
        try:
            traci.vehicle.add(
                vehID=v["veh_id"],
                routeID=route_id,
                typeID=v["type"],
                depart=str(traci.simulation.getTime())
            )
        except traci.TraCIException as e:
            print(f"⚠️ Could not spawn {v['veh_id']}: {e}")

# ▶️ Run a single episode
def run_episode(agent, name, env, vehicle_schedule):
    state = env.reset()
    total_reward = 0
    queue_lengths = []
    waiting_times = []
    done = False
    step = 0

    while not done and step < len(vehicle_schedule):
        # Spawn vehicles for this step
        try:
            spawn_vehicle_plan(vehicle_schedule[step])
        except Exception as e:
            print(f"⚠️ Vehicle spawn failed at step {step}: {e}")

        # Agent takes action
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        # Track stats
        queue_lengths.append(sum(next_state))
        waiting_times.append(env.get_total_waiting_time())
        state = next_state
        total_reward += reward
        step += 1

    throughput = env.get_total_vehicles_exit()
    avg_queue = np.mean(queue_lengths) if queue_lengths else 0
    avg_waiting = np.mean(waiting_times) if waiting_times else 0

    print(f"✅ {name}: Reward={total_reward:.2f}, Throughput={throughput}, Avg Queue={avg_queue:.2f}, Avg Waiting={avg_waiting:.2f}s")
    return {"name": name, "reward": total_reward, "throughput": throughput, "avg_queue": avg_queue, "avg_waiting": avg_waiting}

# 📊 Compare all agents
def compare():
    print("\n🧪 Comparing agents against fixed-time baseline...")
    results = []

    # 🔁 Generate fixed vehicle schedule
    vehicle_schedule = generate_vehicle_schedule(steps=MAX_STEPS, vehicles_per_step=VEHICLES_PER_STEP, seed_value=RANDOM_SEED)

    # Run baseline agent
    env = TrafficEnv("intersection.sumocfg", max_steps=MAX_STEPS, gui=True)
    fixed_agent = FixedTimeAgent(action_size=4, switch_interval=FIXED_INTERVAL, sim_steps_per_decision=40)
    baseline_result = run_episode(fixed_agent, "Fixed-Time", env, vehicle_schedule)

    env.close()
    time.sleep(0.2)

    # Run all RL agents
    for ag in AGENTS:
        env = TrafficEnv("intersection.sumocfg", gui=True, max_steps=MAX_STEPS)
        agent = ag["class"](state_size=28, action_size=4)
        agent.load_model(ag["model_path"])
        agent.epsilon = 0.0
        result = run_episode(agent, ag["name"], env, vehicle_schedule)
        env.close()
        time.sleep(0.2)
        results.append(result)

    # 📈 Show improvements
    print("\n📊 Percentage improvement over Fixed-Time:")
    for r in results:
        r["throughput_improvement"] = ((r["throughput"] - baseline_result["throughput"]) / baseline_result["throughput"]) * 100
        r["queue_reduction"] = ((baseline_result["avg_queue"] - r["avg_queue"]) / baseline_result["avg_queue"]) * 100
        r["waiting_reduction"] = ((baseline_result["avg_waiting"] - r["avg_waiting"]) / baseline_result["avg_waiting"]) * 100
        print(f"{r['name']} -> Throughput: +{r['throughput_improvement']:.1f}%, Queue: -{r['queue_reduction']:.1f}%, Waiting: -{r['waiting_reduction']:.1f}%")

    # 📊 Plot
    names = [r["name"] for r in results]
    tput = [r["throughput_improvement"] for r in results]
    queue = [r["queue_reduction"] for r in results]
    wait = [r["waiting_reduction"] for r in results]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, tput, width, label='Throughput ↑')
    plt.bar(x, queue, width, label='Queue ↓')
    plt.bar(x + width, wait, width, label='Waiting Time ↓')

    plt.ylabel("% Improvement")
    plt.title("Agent vs Fixed-Time Traffic Controller")
    plt.xticks(x, names)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    os.makedirs("comparison", exist_ok=True)
    plt.savefig("comparison/comparison_chart.png")
    print("\n📈 Saved comparison plot to comparison/comparison_chart.png")

# 🏁 Entry point
if __name__ == "__main__":
    compare()
