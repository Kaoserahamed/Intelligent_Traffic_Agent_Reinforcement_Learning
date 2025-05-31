class BaseTrafficAgent:
    def __init__(self):
        self.name = "Base"

    def choose_action(self, state):
        raise NotImplementedError

    def learn(self, state, action, reward, next_state):
        pass

    def set_phase(self, traci, tl_id, action):
        # Default: set traffic light phase directly
        try:
            phase_string = self.get_phase_string(action)
            traci.trafficlight.setRedYellowGreenState(tl_id, phase_string)
        except Exception as e:
            print(f"Error setting phase: {e}")

    def get_phase_string(self, action):
        # Override this to map actions to phase strings
        raise NotImplementedError
