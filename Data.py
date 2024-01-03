import traci
import os

sumo_path = 'sumo_files/osm.sumocfg'

traci.start(["sumo-gui", "-c", sumo_path])

waiting_times = {}
co2_emission = 0
max_step = 200
sum_queue = 0

def get_queue_length(): 
     Halt_N = traci.edge.getLastStepHaltingNumber("-125514711")
     Halt_S = traci.edge.getLastStepHaltingNumber("125514709")
     Halt_E = traci.edge.getLastStepHaltingNumber("548975769")
     Halt_W = traci.edge.getLastStepHaltingNumber("-125514713")
     
     return Halt_N, Halt_S, Halt_E, Halt_W

for step in range(max_step):
     traci.simulationStep()
     #print(step)

     vehicles = traci.vehicle.getIDList()
     for vehicle in vehicles: 

          #Getting waitng time: 
          wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle)
          waiting_times[vehicle] = wait_time

          #Getting CO2 emission: 
          step_length = traci.vehicle.getActionStepLength(vehicle)
          co2_emission_car = traci.vehicle.getCO2Emission(vehicle)

          co2_emission += float(step_length) * co2_emission_car

          #Getting the total halting cars in this step
          get_queue_length()
          sum_queue += sum(get_queue_length())


total_waiting_time = sum(waiting_times.values())          

#Counting vehicles and halting vehicles
total_vehicle_count = 0
halting_count = 0
for value in waiting_times.values():
     total_vehicle_count += 1
     if value != 0:
          halting_count += 1


print("This is the avg. waitingTime: ",total_waiting_time/total_vehicle_count)
print("This is the halting vehicle count: ", halting_count)
print("This is the CO2 Emission: ", round(co2_emission/1000), "gram CO2")
print("This is the avg. halting number per step: ", sum_queue/max_step)

traci.close()