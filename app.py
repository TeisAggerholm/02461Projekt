import importlib
from . import utils
from support.simulation import Simulation

settings = utils.read_settings('settings.json')

# Environment
environment_name = settings['environment']['name']
environment = getattr(importlib.import_module(f'environments.{environment_name.lower()}'), environment_name)() # args mangler

# Model
model_name = utils.read_settings('settings.json')['model']['name']
model = getattr(importlib.import_module(f'models.{model_name.lower()}'), model_name)() # args mangler

# Params
sumo_mode = settings["simulation"]["sumo_mode"]
sumo_path = settings["simulation"]["sumo_path"]
max_step = settings["simulation"]["max_step"]

# Simulation
Simulation(sumo_mode, sumo_path, max_step, environment, model)