import importlib
import utils
from simulation import Simulation
from environments.crossintersection import CrossIntersection
from models.interval_model import Interval_model
from models.dqn import DQN, Memory
import matplotlib.pyplot as plt


# Params
sumo_mode = "sumo"
max_step = 200
percentage_straight = 0.5
min_green_phase_steps = 10
yellow_phase_steps = 2
red_phase_steps = 2
final_score_weights = {"total_waiting_time": 1, "halting_vehicle_count": 1,
                       "co2_emission_total": 1, "avg_queue_length": 1, 
                       "avg_wait_time": 1}
car_intensity_per_min = 15
spredning = 15

# Environment
environment = CrossIntersection(sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps, max_step, percentage_straight, car_intensity_per_min, spredning) 

# DQN Model
input_dim = 4
hidden_dim = 124
epsilon_decrease = 0.01**(1/10000) # 0.01 fjernes hver 10000 gang.
gamma = 0.99
model = DQN(environment.num_actions, input_dim, hidden_dim, epsilon_decrease, gamma)
memory = Memory(10000)

# Interval_model
interval = 15
#model = Interval_model(environment.num_actions, interval, yellow_phase_steps, red_phase_steps)

# Simulation
episodes = 1000

episode_stats = []

# Initialize a plot
plt.figure(figsize=(10, 6))
plt.ion()  # Turn on interactive mode

for episode in range(episodes):
    print(f"-----------------------------Simulating episode {episode+1}-----------------------------")
    # Assuming Simulation is defined elsewhere
    simulation = Simulation(max_step, environment, model, final_score_weights, episodes, memory)
    simulation.run()
    print("Overall reward: ", simulation.overall_reward)
    episode_stats.append(simulation.overall_reward)

    # Generating a list of episode indices
    episode_indices = list(range(1, episodes + 1))

    # Plotting the data so far
    plt.scatter(episode + 1, simulation.overall_reward)  # Plot the new data point
    plt.title('Overall Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Overall Reward')
    plt.xlim(1, episodes)  # Set the limit for x-axis
    plt.ylim(min(episode_stats) - 10, max(episode_stats) + 10)  # Set the limit for y-axis dynamically
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)  # Pause to update the plot

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot