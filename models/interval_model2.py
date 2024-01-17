class Interval_model:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.last_action_index = -1
        self.last_action_step = -1

    def choose_action(self, state, current_step):
        self.last_action_index = (self.last_action_index + 1) % self.num_actions
        return self.last_action_index

    def train(self, batch):
        pass
    def save_model(self):
        pass
    