import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, num_actions, state_dim, hidden_dim, epsilon_decrease, gamma, weights_path):
        super(DQN, self).__init__()
        self.epsilon = 0
        self.epsilon_decrease = epsilon_decrease
        self.gamma = gamma
        self.learning_rate = 0.001

        self.num_actions = num_actions
        action_dim = num_actions
        self.weights_path = weights_path
    
        # input layer, Activation layer (ReLU), Output layer
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, action_dim),
        )

        try: 
            self.load_model()
            print("----Weights loaded------")

        except FileNotFoundError:
            print("---- No pretrained model found -----")


        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, state):
        return self.net(state)

    def get_qs(self, states):
        qs = []
        for state in states:
            Q_values = torch.zeros(2)
            for i in range(2):
                state = torch.cat((state, torch.Tensor([i])))
                Q_values[i] = self.net(state)
                state = state[:-1]
            qs.append(torch.max(Q_values, dim=0).values)
        return torch.tensor(qs)

    def choose_action(self, state, currentStep):
        if random.random() < self.epsilon: 
            action = random.choice(range(2)) #Vælger random action 0 eller 1. 
    
        else:
            Q_values = torch.zeros(2)
            for i in range(2):
                state = torch.cat((state, torch.Tensor([i])))
                Q_values[i] = self.net(state)
                state = state[:-1]
            action = torch.argmax(Q_values).item()
        return action

    def train(self, batch):
        if batch is None:
            return None
        
        obs_buffer, action_buffer, reward_buffer, obs_next_buffer, done_buffer = zip(*batch)
        states = torch.Tensor(obs_buffer)
        actions = torch.LongTensor(action_buffer)
        rewards = torch.Tensor(reward_buffer)
        next_states = torch.Tensor(obs_next_buffer)
        dones = torch.Tensor(done_buffer)
        
        Q_values = self.net(torch.concat((states, actions.reshape((-1,1))), dim=1))
        next_Q_values = self.get_qs(next_states)

        target_Q_values = rewards + self.gamma * next_Q_values * (1 - dones)
        loss = self.loss_function(Q_values, target_Q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_weight_mean = sum(p.data.mean() for p in self.net.parameters()) / len(list(self.net.parameters()))
        #print(f"-----Average weight mean------: {total_weight_mean}")

        return loss.item()
    
    def epsilon_dec_fun(self, currentStep): 
        
        if self.epsilon == 0: 
            print(self.epsilon)
            return 
        self.epsilon = (self.epsilon - 0.1) * self.epsilon_decrease + 0.1
        print("Epsilon: ",self.epsilon)

    def save_model(self):
        torch.save(self.state_dict(), "weights/"+self.weights_path)
    
    def load_model(self):
        self.load_state_dict(torch.load("weights/"+self.weights_path))
