class Interval_model:
    def __init__(self, num_actions, interval, yellow_phase_steps, red_phase_steps):
        self.interval = interval
        self.num_actions = num_actions
        self.last_action_index = -1
        self.last_action_step = -1
        self.yellow_phase_steps = yellow_phase_steps
        self.red_phase_steps = red_phase_steps
        self.epsilon = 0

    def choose_action(self, state_list, currentStep):
        
        if currentStep == 0:
            return self._change_action(currentStep)
        elif currentStep - self.last_action_step >= self.interval and currentStep == self.interval:
            return self._change_action(currentStep)
        elif currentStep - self.last_action_step - self.yellow_phase_steps - self.red_phase_steps >= self.interval:
            return self._change_action(currentStep)
        else:
            return self.last_action_index
    
    def _change_action(self, currentStep):
        self.last_action_step = currentStep
        self.last_action_index = (self.last_action_index + 1) % self.num_actions
        return self.last_action_index


    def train(self, batch):
        pass

    def save_model(self, pth):
        pass
    def epsilon_dec_fun(self):
        pass