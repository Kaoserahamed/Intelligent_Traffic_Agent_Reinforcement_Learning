# fixed_agent.py
class FixedTimeAgent:
    def __init__(self, action_size, switch_interval=20, sim_steps_per_decision=30):
        self.action_size = action_size
        self.switch_interval = switch_interval
        self.sim_steps_per_decision = sim_steps_per_decision
        self.counter = 0

    def choose_action(self, state):
        action = (self.counter // self.switch_interval) % self.action_size
        self.counter += 1
        return action
