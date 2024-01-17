import importlib
import utils
from simulation3 import Simulation
# from simulation import Simulation
# from environments.crossintersection import CrossIntersection
from environments.crossintersection2 import CrossIntersection
from models.interval_model2 import Interval_model
from models.dqn import DQN
from models.dqn_old import DQN_prev
from memory import Memory
import matplotlib.pyplot as plt
import csv 
import os 
import time

# Environment
sumo_mode = "sumo"
max_step = 500
percentage_straight = 0.75
min_green_phase_steps = 10
yellow_phase_steps = 2
red_phase_steps = 1
car_intensity_per_min = 15
spredning = 7
seed = True
environment = CrossIntersection(sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps, max_step, seed) 

# DQN Model
hidden_dim = 124
epsilon_decrease = 0.01**(1/1000) # 0.1 fjernes pr. 100 epsioder
gamma = 0.99
weights_path = "17.01v5.pth"
#model = DQN(1, 5, hidden_dim, epsilon_decrease, gamma, weights_path)
memory = Memory(50000)

# model = DQN_prev(2,80,hidden_dim,epsilon_decrease,gamma,weights_path)

# Interval_model
model = Interval_model(environment.num_actions)

# Simulation
overall_reward_queue_length = []
ephocs = 200
batch_size = 100

#SAVE TO CSV file: 
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

model_class_name = type(model).__name__
num_files = len([name for name in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, name))])

file_name = f"rewards_{model_class_name}_{num_files}.csv"
file_path = os.path.join(data_folder, file_name)

rewards = []
loss = []

# LOOP SETTINGS
episodes = 2
x_hours = 0
end_time = time.time() + x_hours * 3600

episode = 0

while time.time() < end_time or len(rewards) < episodes:
    episode += 1
    print(f"-----------------------------Simulating episode {episode}-----------------------------")
    simulation = Simulation(max_step, environment, model, memory, ephocs, batch_size)
    simulation._model.epsilon = 1 - (episode/ episodes)

    simulation.run()
    print("Overall reward: ", simulation.overall_reward)

    loss.extend(simulation.episode_losses)
    rewards.append(simulation.overall_reward)

with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(rewards)):
        writer.writerow([rewards[i]])


model.save_model()
print('Model saved')