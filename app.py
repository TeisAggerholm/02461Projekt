import importlib
import utils
from simulation import Simulation
from environments.crossintersection import CrossIntersection
from models.interval_model import Interval_model
from models.dqn import DQN

# Params
sumo_mode = "sumo"
max_step = 300
percentage_straight = 0.5
min_green_phase_steps = 10
yellow_phase_steps = 2
red_phase_steps = 2
final_score_weights = {"total_waiting_time": 1, "halting_vehicle_count": 1,
                       "co2_emission_total": 1, "avg_queue_length": 1, 
                       "avg_wait_time": 1}
car_intensity_per_min = 10
spredning = 15

# Environment
environment = CrossIntersection(sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps, max_step, percentage_straight, car_intensity_per_min, spredning) 

# Model
input_dim = 4
hidden_dim = 124
epsilon_decrease = 0.1**(1/1000) # 0.1 fjernes hver 1000 gang.
gamma = 0.99
model = DQN(environment.num_actions, input_dim, hidden_dim, epsilon_decrease, gamma)

# Simulation
episodes = 1
for episode in range(episodes):
    print(f"-----------------------------Simulating episode {episode+1}-----------------------------")
    simulation = Simulation(max_step, environment, model, final_score_weights, episodes)
    simulation.run()
    #print('-------VORES REGNEDE STATS-------:', simulation.stats)
    print(simulation.memory._experiences[100])
    print("Overall reward: ", simulation.overall_reward)

