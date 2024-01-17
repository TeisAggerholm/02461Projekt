import importlib
import utils
from simulation import Simulation
from environments.crossintersection import CrossIntersection
from models.interval_model import Interval_model
from models.dqn import DQN
from models.dqn_old import DQN_prev
from memory import Memory
import matplotlib.pyplot as plt
import csv 
import os 

# Environment
sumo_mode = "sumo"
max_step = 500
percentage_straight = 0.75
min_green_phase_steps = 10
yellow_phase_steps = 2
red_phase_steps = 2
car_intensity_per_min = 15
spredning = 7
seed = None
final_score_weights = {"total_waiting_time": 1, "halting_vehicle_count": 1,
                       "co2_emission_total": 1, "avg_queue_length": 1, 
                       "avg_wait_time": 1}
environment = CrossIntersection(sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps, max_step, percentage_straight, car_intensity_per_min, spredning, seed) 

# DQN Model
hidden_dim = 100
epsilon_decrease = 0.01**(1/1000) # 0.1 fjernes pr. 100 epsioder
gamma = 0.99
weights_path = "assasaasada.pth"
#model = DQN(1, 5, hidden_dim, epsilon_decrease, gamma, weights_path)
memory = Memory(50000)

model = DQN_prev(2,80,hidden_dim,epsilon_decrease,gamma,weights_path)

# Interval_model
interval = 35
#model = Interval_model(environment.num_actions, interval, yellow_phase_steps, red_phase_steps)

# Simulation
episodes = 500
episode_stats = []
overall_reward_queue_length = []
ephocs = 400

# Initialize a plot
plt.figure(figsize=(12, 10))
plt.ion()

#SAVE TO CSV file: 
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

model_class_name = type(model).__name__
num_files = len([name for name in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, name))])

file_name = f"rewards_{model_class_name}_{num_files}.csv"
file_path = os.path.join(data_folder, file_name)

rewards = []
queue_length = []
loss = []

loss = []


for episode in range(episodes):
    print(f"-----------------------------Simulating episode {episode+1}-----------------------------")
    # Assuming Simulation is defined elsewhere
    simulation = Simulation(max_step, environment, model, final_score_weights, episodes, memory, ephocs)
        
    simulation._model.epsilon = max(1 - (episode / episodes)*2, 0.1)

    simulation.run()
    print("Overall reward: ", simulation.overall_reward)
    episode_stats.append(simulation.overall_reward)
    overall_reward_queue_length.append(simulation.overall_reward_queue_length)
    loss.extend(simulation.episode_losses)

    # Generating a list of episode indices
    episode_indices = list(range(1, episodes + 1))

    # First subplot for overall_reward
    plt.subplot(2, 1, 1)  # (rows, columns, panel number)
    plt.scatter(episode + 1, simulation.overall_reward, color='b')
    plt.title(f'Overall Reward per Episode\nEpsilon: {simulation._model.epsilon}')
    plt.xlabel('Episode')
    plt.ylabel('Overall Reward')
    plt.xlim(1, episodes)
    plt.ylim(min(episode_stats) - 10, max(episode_stats) + 10)
    plt.grid(True)

    # Second subplot for overall_reward_queue_length
    plt.subplot(2, 1, 2)
    plt.scatter(episode + 1, overall_reward_queue_length[-1], color='r')
    plt.title('Overall Reward Queue Length per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward Queue Length')
    plt.xlim(1, episodes)
    plt.ylim(min(overall_reward_queue_length) - 10, max(overall_reward_queue_length) + 10)
    plt.grid(True)

    plt.pause(0.1)  # Pause to update the plots


    #Save overall reward to CSV.
    rewards.append(simulation.overall_reward)
    queue_length.append(simulation.overall_reward_queue_length)
    print(simulation.overall_reward_queue_length)

with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(rewards)):
        writer.writerow([rewards[i], queue_length[i]])
    

model.save_model()
print('Model saved')

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot
