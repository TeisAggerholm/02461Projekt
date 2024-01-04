import traci
import os

class simulation:
     waiting_times = {}
     co2_emission = 0
     sum_queue = 0
     def __init__(self, sumo_path, max_step, gui:bool):
          self.sumo_path = sumo_path
          self.max_step = max_step
          self.gui = gui     

     def run(self):
          sim = simulation(self.sumo_path,self.max_step,self.gui)

          if self.gui: 
               sumo = "sumo-gui"
          else: 
               sumo = "sumo"
     
          traci.start([sumo, "-c", self.sumo_path])
          for step in range(self.max_step):
               traci.simulationStep()
               vehicles = traci.vehicle.getIDList()
               for vehicle in vehicles:
                    sim.get_waiting_time(vehicle)
                    self.co2_emission += sim.get_co2_emission(vehicle)

                    self.sum_queue += sum(sim.get_queue_length())
          
          total_waiting_time = sum(self.waiting_times.values())

          avg_wait_time = total_waiting_time/sim.count_halting_vehicles()[0]
          halting_vehicle_count = sim.count_halting_vehicles()[1]
          co2_emission_total = round(self.co2_emission/1000)
          avg_queue_length = self.sum_queue/self.max_step

          traci.close()

          return avg_wait_time, halting_vehicle_count, co2_emission_total, avg_queue_length

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
     
if __name__ == '__main__':
     print("These are the total stats of the run")
     print(simulation('sumo_files/osm.sumocfg', 200,False).run())
