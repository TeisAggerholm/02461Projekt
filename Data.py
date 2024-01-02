import traci

traci.start(["sumo", "-c", "/Users/hectorheltjakobsen/sumo/tools/2024-01-02-14-05-12/osm.sumocfg"])


for step in range(10):
    traci.simulationStep()

    vehicle_positions = traci.vehicle.getPosition("vehicle_id")
    vehicle_speed = traci.vehicle.getSpeed("vehicle_id")

traci.close()