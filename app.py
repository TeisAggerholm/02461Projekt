import importlib
from simulation import Simulation
from environments.crossintersection import CrossIntersection
from models.interval_model import Interval_model
from models.dqn import DQN
from memory import Memory
import matplotlib.pyplot as plt
import csv 
import os 
import time

# Environment
sumo_mode = "sumo"
max_step = 500
min_green_phase_steps = 10
yellow_phase_steps = 2
red_phase_steps = 1

environment = CrossIntersection(sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps, max_step) 

# Model parameters
hidden_dim = 124
gamma = 0.75
weights_path = "Training_overnight.pth"
memory = Memory(50000)


# Dqn_model
model = DQN(2,80,hidden_dim,gamma,weights_path)

# Interval_model
# model = Interval_model(environment.num_actions)

# Simulation
ephocs = 200
batch_size = 100

# LOOP SETTINGS
episodes = 160
x_hours = 0
end_time = time.time() + x_hours * 3600

rewards = []
loss = []

episode = 0
while time.time() < end_time or len(rewards) < episodes:
    episode += 1
    print(f"-----------------------------Simulating episode {episode}-----------------------------")
    simulation = Simulation(max_step, environment, model, memory, ephocs, batch_size)
    simulation._model.epsilon = max(1 - (episode/episodes),0.1)
    print(f"Epsilon: {simulation._model.epsilon}")
    #simulation._model.epsilon = 0
    simulation.run()
    print("Overall reward: ", simulation.overall_reward)

    loss.extend(simulation.episode_losses)
    rewards.append(simulation.overall_reward)

    #Temp model save
    if episode % 100 == 0: 
        model.save_model()


# SAVE TO CSV file: 
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

model_class_name = type(model).__name__
num_files = len([name for name in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, name))])

file_name = f"rewards_{model_class_name}_{num_files}.csv"
file_path = os.path.join(data_folder, file_name)

with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(rewards)):
        writer.writerow([rewards[i]])


# SAVE MODEL
model.save_model()
print('Model saved final')