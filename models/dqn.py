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
        self.epsilon = 0
        self.epsilon_decrease = epsilon_decrease
        self.gamma = gamma
        self.learning_rate = 0.001

        self.num_actions = num_actions
        action_dim = num_actions

        # input layer, Activation layer (ReLU), Output layer
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax()
        )
        try: 
            self.load_model("test1.pth")
            print("----Weights loaded------")

        except FileNotFoundError:
            print("---- No pretrained model found -----")

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_function = torch.nn.MSELoss()

    def forward(self, state):
        return self.net(state)

    def choose_action(self, state_list, currentStep):
        if random.random() < self.epsilon: 
            action = random.choice(range(self.num_actions))
        else: 
            state = self.convert_to_tensor(state_list)
            print(state)
            action = torch.argmax(self.net(state)).item()
            print(action)
        # print("-----TENSOR------", (self.net(state).detach().numpy()))
        # print("------ACTION------", action)    
        return action
    
    def convert_to_tensor(self, state_list):
        return torch.Tensor(state_list)

    def train(self, batch):
        if batch is None:
            return None
        
        obs_buffer, action_buffer, reward_buffer, obs_next_buffer, done_buffer = zip(*batch)
        states = torch.Tensor(obs_buffer)
        actions = torch.LongTensor(action_buffer)
        rewards = torch.Tensor(reward_buffer)
        next_states = torch.Tensor(obs_next_buffer)
        dones = torch.Tensor(done_buffer)

        #State action pair som en vektor til input til det neurale netværk. 

        Q_values = self.net(states).gather(1, actions.unsqueeze(1))
        
        #Dette gør at der ikke bliver taget gradients af next state.
        with torch.no_grad():
            next_Q_values = self.net(next_states).max(1)[0].detach()
            target_Q_values = rewards + self.gamma * next_Q_values * (1 - dones)
        
        loss = self.loss_function(Q_values, target_Q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_weight_mean = sum(p.data.mean() for p in self.net.parameters()) / len(list(self.net.parameters()))
        print(f"-----Average weight mean------: {total_weight_mean}")

        return loss.item()

    def epsilon_dec_fun(self): 
        self.epsilon = (self.epsilon - 0.1) * self.epsilon_decrease + 0.1
        print("Epsilon: ",self.epsilon)

    def save_model(self, file_name):
        torch.save(self.state_dict(), file_name)
    
    def load_model(self, file_name):
        self.load_state_dict(torch.load(file_name))


#Loss function 


# Reward
# Loss function
# Batch, memory læring

# Batch state
# Batch reward
# Batch action
# Batch next state 
