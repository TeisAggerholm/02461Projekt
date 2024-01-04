import time

class Interval_model:
    def __init__(self, num_actions, interval):
        self.interval = interval
        self.num_actions = num_actions
        self.last_action_time = time.time()
        self.last_action_index = -1

    def choose_action(self, state, *args):
        current_time = time.time()
        if current_time - self.last_action_time >= self.interval:
            self.last_action_time = current_time
            self.last_action_index = (self.last_action_index + 1) % self.num_actions
            return self.last_action_index
        else:
            return None