import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class Reward:
    def __init__(self, statistics : dict, weights : dict):
        self.statistics = statistics
        self.weights = weights
    
    def calculate(self):
        reward = None

        reward = self.weights["wait_times"] * self._convert_wait_times(self.statistics["wait_times"]) + self.weights["total_co2"] * self._convert_co2(self.statistics["total_co2"]) + self.weights["queues"] * self._convert_queues(self.statistics["queues"])
        return reward
    
    def _convert_wait_times(self, wait_times):
        return sum(wait_times/len(wait_times))

    def _convert_co2(self, total_co2):
        return total_co2 / len(self.statistics["wait_time"])
    
    def _convert_queues(self, queues):
        return sum(queues)/len(self.statistics["wait_time"])

class DQN(nn.Module):
    def __init__(self, num_actions, state_dim, hidden_dim):
        super(DQN, self).__init__()
        
        self.num_actions = num_actions
        action_dim = num_actions
    
        # input layer, Activation layer (ReLU), Output layer
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(self, state):
        return self.net(state)


    def choose_action(self, state_description):
        if random.random() < 0.1: 
            action = random.choice(range(self.num_actions))
        else: 
            state = self.convert_to_tensor(state_description["statistics"])
            action = torch.argmax(self.net(state)).item()

        print("------ACTION------", action)    
        return action
    
    def convert_to_tensor(self, statistics):
        if not statistics["wait_times"]: 
            avg_wait = 0
            total_co2 = 0 
        else: 
            avg_wait = sum(statistics["wait_times"])/len(statistics["wait_times"])
            total_co2 = statistics["total_co2"]/len(statistics["wait_times"])

        queue_N = statistics["queues"][0]
        queue_S = statistics["queues"][1]
        queue_E = statistics["queues"][2]
        queue_W = statistics["queues"][3]

        return torch.Tensor([avg_wait,total_co2,queue_N,queue_S,queue_E,queue_W])





#Loss function 
Loss_fn = torch.nn.MSELoss(reduction='sum')

