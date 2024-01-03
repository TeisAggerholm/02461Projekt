import traci
import os


sumo_config_path = 'sumo_files/osm.sumocfg'

traci.start(["sumo", "-c", sumo_config_path])

for step in range(200):
     traci.simulationStep()
     print(step)


traci.close()