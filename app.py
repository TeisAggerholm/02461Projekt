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

# Simulation
Simulation()