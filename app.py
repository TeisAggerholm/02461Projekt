import importlib
import utils
from support.simulation import Simulation

settings = utils.read_settings('settings.json')

# Params
sumo_mode = settings["simulation"]["sumo_mode"]
max_step = settings["simulation"]["max_step"]
min_green_phase_steps = settings["environment"]["min_green_phase_steps"]
yellow_phase_steps = settings["environment"]["yellow_phase_steps"]
red_phase_steps = settings["environment"]["red_phase_steps"]
percentage_straight = settings["simulation"]["percentage_straight"]

# Environment
environment_name = settings['environment']['name']
environment_class = getattr(importlib.import_module(f'environments.{environment_name.lower()}'), environment_name)
environment = environment_class(sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps, max_step, percentage_straight) 

# Model
model_args = settings["model"]["*args"]
model_name = settings['model']['name']
model_class = getattr(importlib.import_module(f'models.{model_name.lower()}'), model_name)
model = model_class(environment.num_actions, *model_args)

# Simulation
simulation = Simulation(max_step, environment, model)
simulation.run()
print('-------VORES REGNEDE STATS-------:', simulation.stats)