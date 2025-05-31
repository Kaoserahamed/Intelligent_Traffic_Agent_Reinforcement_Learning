from agents.dqn_agent import DQNTrafficAgent
import numpy as np

class DoubleDQNTrafficAgent(DQNTrafficAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_model = self._build_model(kwargs.get("learning_rate", 0.001))
        self.name = "Double_DQN"
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update(self, state, action, reward, next_state):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])

        action_next = np.argmax(self.model.predict(next_state, verbose=0)[0])
        target = reward + self.gamma * self.target_model.predict(next_state, verbose=0)[0][action_next]

        target_f = self.model.predict(state, verbose=0)
        target_f[0][action] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)
        self.update_target_model()
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay