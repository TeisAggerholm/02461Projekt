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

# Environment
environment_name = settings['environment']['name']
environment = getattr(importlib.import_module(f'environments.{environment_name.lower()}'), environment_name)(sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps) 

# Model
model_args = settings["model"]["*args"]
model_name = utils.read_settings('settings.json')['model']['name']
model = getattr(importlib.import_module(f'models.{model_name.lower()}'), model_name)(environment.num_actions, *model_name)

# Simulation
Simulation(max_step, environment, model)