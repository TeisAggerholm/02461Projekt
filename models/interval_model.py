class Interval_model:
    def __init__(self, num_actions, interval):
        self.interval = interval
        self.num_actions = num_actions
        self.last_action_index = -1
        self.last_action_step = -1

    def choose_action(self, state):
        current_step = state["currentStep"]
        if current_step - self.last_action_step >= self.interval or current_step == 0:
            self.last_action_step = current_step
            self.last_action_index = (self.last_action_index + 1) % self.num_actions
            return self.last_action_index
        else:
            return self.last_action_index