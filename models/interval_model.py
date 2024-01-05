class Interval_model:
    def __init__(self, num_actions, interval):
        self.interval = interval
        self.num_actions = num_actions
        self.last_action_index = -1
        self.last_action_step = -1

    def choose_action(self, state):
        current_step = state["currentStep"]

        print('currentStep:', current_step)
        if current_step - self.last_action_step >= self.interval or current_step == 0:
            self.last_action_step = current_step

            print('action_old:', self.last_action_index)
            self.last_action_index = (self.last_action_index + 1) % self.num_actions
            print('action_new:', self.last_action_index)
            return self.last_action_index
        else:
            return self.last_action_index