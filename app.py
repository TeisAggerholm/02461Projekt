import importlib
import utils
from simulation import Simulation
from environments.crossintersection import CrossIntersection
from models.interval_model import Interval_model
from models.dqn import DQN, Memory
import matplotlib.pyplot as plt


# Params
sumo_mode = "sumo-gui"
max_step = 400
percentage_straight = 0.75
min_green_phase_steps = 10
yellow_phase_steps = 2
red_phase_steps = 2
final_score_weights = {"total_waiting_time": 1, "halting_vehicle_count": 1,
                       "co2_emission_total": 1, "avg_queue_length": 1, 
                       "avg_wait_time": 1}
car_intensity_per_min = 15
spredning = 7
seed = None

# Environment
environment = CrossIntersection(sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps, max_step, percentage_straight, car_intensity_per_min, spredning, seed) 

# DQN Model
input_dim = 5
hidden_dim = 200
epsilon_decrease = 0.01**(1/1000) # 0.1 fjernes pr. 100 epsioder
gamma = 0.99
model = DQN(1, input_dim, hidden_dim, epsilon_decrease, gamma)
memory = Memory(50000)

# Interval_model
interval = 15
#model = Interval_model(environment.num_actions, interval, yellow_phase_steps, red_phase_steps)

# Simulation
episodes = 200

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
    plt.title(f'Overall Reward per Episode. Epsilon: {round(simulation._model.epsilon,2)}')
    plt.xlabel('Episode')
    plt.ylabel('Overall Reward epsilon')
    plt.xlim(1, episode)
    plt.ylim(min(episode_stats) - 10, max(episode_stats) + 10)  # Set the limit for y-axis dynamically
    plt.grid(True)
    plt.draw()
    plt.pause(0.1)  # Pause to update the plot

model.save_model("Newreward2.pth")

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot
