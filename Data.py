import traci
import os

class simulation: 
     def run(self, sumo_path): 
          self.sumo_path = sumo_path

          traci.start(["sumo", "-c", sumo_path])

          max_step = 200
          for step in range(max_step):
               traci.simulationStep()
               print(step)

          self.waiting_times = {}
          current_waiting_time = self.collecting_waiting_times(self)

          traci.close()


     def collecting_waiting_times(self): 
          vehicles = traci.vehicle.getIDList()
          for vehicle in vehicles: 
               wait_time = traci.vehicle.getAccumulatedWaitingTime(vehicle)
          
               self.waiting_times[vehicle] = wait_time
               
               if vehicle in self._waiting_times: #GITHUB guy, ved ikke hvorfor men noget med at bilen bliver slettet, ved clearer intersection
                         del self._waiting_times[vehicle]

          total_waiting_time = sum(self._waiting_times.values())

          return total_waiting_time

simulation.run('sumo_files/osm.sumocfg')