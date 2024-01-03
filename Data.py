import traci
import os

# sumo_path = 'sumo_files/osm.sumocfg'

class simulation:
     waiting_times = {}
     co2_emission = 0
     sum_queue = 0
     def __init__(self, sumo_path, max_step):
          self.sumo_path = sumo_path
          self.max_step = max_step
     
     def run(self):
          traci.start(["sumo", "-c", self.sumo_path])
          for step in range(self.max_step):
               traci.simulationStep()
               vehicles = traci.vehicle.getIDList()
               for vehicle in vehicles:
                    simulation(self.sumo_path,self.max_step).get_waiting_time(vehicle)
                    self.co2_emission += simulation(self.sumo_path,self.max_step).get_co2_emission(vehicle)

                    self.sum_queue += sum(simulation(self.sumo_path,self.max_step).get_queue_length())
          
          total_waiting_time = sum(self.waiting_times.values())
      
          print("This is the avg. waitingTime: ",total_waiting_time/simulation(self.sumo_path,self.max_step).count_halting_vehicles()[0])
          print("This is the halting vehicle count: ", simulation(self.sumo_path,self.max_step).count_halting_vehicles()[1])
          print("This is the CO2 Emission: ", round(self.co2_emission/1000), "gram CO2")
          print("This is the avg. halting number per step: ", self.sum_queue/self.max_step)

          traci.close()

     def get_queue_length(self): 
          Halt_N = traci.edge.getLastStepHaltingNumber("-125514711")
          Halt_S = traci.edge.getLastStepHaltingNumber("125514709")
          Halt_E = traci.edge.getLastStepHaltingNumber("548975769")
          Halt_W = traci.edge.getLastStepHaltingNumber("-125514713")
          
          return Halt_N, Halt_S, Halt_E, Halt_W

     def get_waiting_time(self, vehicle):
          self.vehicle = vehicle
          wait_time = traci.vehicle.getAccumulatedWaitingTime(self.vehicle)
          self.waiting_times[self.vehicle] = wait_time
     
     def get_co2_emission(self,vehicle):
          self.vehicle = vehicle
          co2_emission_car = traci.vehicle.getCO2Emission(self.vehicle)
          return co2_emission_car
          

     def count_halting_vehicles(self):
          total_vehicle_count = 0
          halting_count = 0
          for value in self.waiting_times.values():
               total_vehicle_count += 1
               if value != 0:
                    halting_count += 1
          return total_vehicle_count, halting_count

simulation('sumo_files/osm.sumocfg', 200).run()
