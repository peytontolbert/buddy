import pickle
class StateManager:
    def __init__(self):
        pass

    def save_agent_state(self, agent, filename='finnagi_state.pickle'):
        with open(filename, 'wb') as f:
            pickle.dump(agent, f)

    def load_agent_state(self, filename='finnagi_state.pickle'):
        try:
            with open(filename, 'rb') as f:
                loaded_agent = pickle.load(f)
            return loaded_agent
        except FileNotFoundError:
            print("No saved state file found. Creating a new agent.")
