import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self._experiences = []
    
    def add_experience(self, experience):
        self._experiences.append(experience)

        if len(self._experiences) > self.capacity:
            self._experiences.pop(0)

    def get_batch(self, batch_size):
        if len(self._experiences) < batch_size:
            return None  # Not enough samples to create a batch
        return random.sample(self._experiences, batch_size)
        
class Experience:
    def __init__(self, state, action, reward, next_state):
        self.state = state 
        self.action = action
        self.reward = reward 
        self.next_state = next_state

# Batch state
# Batch reward
# Batch action
# Batch next state 

class DQN(nn.Module):
    def __init__(self, num_actions, state_dim, hidden_dim, epsilon_decrease, gamma):
        super(DQN, self).__init__()
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.gamma = gamma
        self.learning_rate = 0.0001

        self.num_actions = num_actions
        action_dim = num_actions
    
        # input layer, Activation layer (ReLU), Output layer
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, action_dim),
        )
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, state):
        return self.net(state)

    def choose_action(self, state_list, currentStep):
        if random.random() < 0.1: 
            action = random.choice(range(self.num_actions))
        else: 
            state = self.convert_to_tensor(state_list)
            action = torch.argmax(self.net(state)).item()

            print("-----TENSOR------", (self.net(state).detach().numpy()))
        print("------ACTION------", action)    
        return action
    
    def convert_to_tensor(self, state_list):
        return torch.Tensor(state_list)

    def train(self, batch):
        pass
        # index 0 = state_list
        # batch_array = np.array(batch)
        # states  = batch_array[:, 0]
        # #  >> via. list comprehension: states = [sublist[0] for sublist in batch if sublist]
        # print(states)        
        # out = self.net(self.convert_to_tensor(states))

        # print(batch)
        


#Loss function 
Loss_fn = torch.nn.MSELoss(reduction='sum')

# Reward
# Loss function
# Batch, memory læring

# Batch state
# Batch reward
# Batch action
# Batch next state 
