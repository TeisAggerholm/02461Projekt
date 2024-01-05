import importlib
import utils
from support.simulation import Simulation
from support.route_generator import generator

settings = utils.read_settings('settings.json')

# Params
sumo_mode = settings["simulation"]["sumo_mode"]
max_step = settings["simulation"]["max_step"]
percentage_straight = settings["simulation"]["percentage_straight"]
min_green_phase_steps = settings["environment"]["min_green_phase_steps"]
yellow_phase_steps = settings["environment"]["yellow_phase_steps"]
red_phase_steps = settings["environment"]["red_phase_steps"]
final_score_weights = settings["simulation"]["total_score_weight_factors"]

# Environment
environment_name = settings['environment']['name']
environment_class = getattr(importlib.import_module(f'environments.{environment_name.lower()}'), environment_name)
environment = environment_class(sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps) 

# Model
model_args = settings["model"]["*args"]
model_name = settings['model']['name']
model_class = getattr(importlib.import_module(f'models.{model_name.lower()}'), model_name)
model = model_class(environment.num_actions, *model_args)

# Simulation
generator(max_step).route_generator(percentage_straight)
simulation = Simulation(max_step, environment, model, final_score_weights)
simulation.run()


print('-------VORES REGNEDE STATS-------:', simulation.stats)
print('overall score', simulation.calc_overall_score())