import traci
import os

sumo_path = 'sumo_files/osm.sumocfg'

traci.start(["sumo", "-c", sumo_path])

waiting_times = {}
max_step = 200
for step in range(max_step):
     traci.simulationStep()
     print(step)

     vehicles = traci.vehicle.getIDList()
     for vehicle in vehicles: 
          wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle)
          print(wait_time)
          
          waiting_times[vehicle] = wait_time


total_waiting_time = sum(waiting_times.values())
print(waiting_times)
print("This is the avg. waitingTime: ",total_waiting_time/29)

traci.close()