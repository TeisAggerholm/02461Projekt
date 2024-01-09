import importlib
import utils
from support.simulation import Simulation
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
car_intensity_per_min = 30
spredning = 15

# Environment
environment = CrossIntersection(sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps, max_step, percentage_straight, car_intensity_per_min, spredning) 

# Model
model = DQN(environment.num_actions, 6, 124)

# Simulation
simulation = Simulation(max_step, environment, model, final_score_weights)
simulation.run()


print('-------VORES REGNEDE STATS-------:', simulation.stats)
print('overall score', simulation.calc_overall_score())
