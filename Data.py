import traci
import os


sumo_config_path = 'sumo_files/2024-01-02-14-05-12/osm.sumocfg'

traci.start(["sumo", "-c", sumo_config_path])

for step in range(10):
    traci.simulationStep()

    vehicle_positions = traci.vehicle.getPosition("vehicle_id")
    vehicle_speed = traci.vehicle.getSpeed("vehicle_id")

traci.close()