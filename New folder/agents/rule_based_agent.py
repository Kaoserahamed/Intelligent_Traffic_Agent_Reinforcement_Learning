from agents.base_agent import BaseTrafficAgent

class RuleBasedAgent(BaseTrafficAgent):
    def __init__(self, action_size):
        self.current = -1
        self.action_size = action_size
        super().__init__()
        self.name = "Rule_Based"

    def choose_action(self, state):
        self.current = (self.current + 1) % self.action_size
        return self.current

    def update(self, state, action, reward, next_state):
        pass