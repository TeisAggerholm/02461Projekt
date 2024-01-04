import traci
import os

class Simulation:
    def __init__(self, sumo_mode, sumo_path, max_step, _environment, _model):
        self.sumo_mode = sumo_mode
        self.sumo_path = sumo_path
        self.max_step = max_step
        self._environment = _environment
        self._model = _model
        self.sum_queue = 0
        self.co2_emission = 0
        self.waiting_times = {}
        self.stats = {}

    def run(self):

        # fremfor traci.start - så self._environment.start() - så har environment selv styr over sumo_path
        traci.start([self.sumo_mode, "-c", self.sumo_path])


        old_action = -1

        for step in range(self.max_step):
            
            # Model + Environment
            state = self._environment.get_state()
            action = self._model.choose_action(state)

            if action != old_action:
                steps_to_do = self._environment.change_phases(action)

                while steps_to_do > 0:
                    traci.simulationStep()
                    steps_to_do -= 1
            else:
                traci.simulationStep()

            
            
            # STATS
            vehicles = traci.vehicle.getIDList()
            for vehicle in vehicles:
                self.get_waiting_time(vehicle)
                self.co2_emission += self.get_co2_emission(vehicle)
                self.sum_queue += sum(self.get_queue_length())

            # UPDATE
            old_action = action

        self.set_stats()
        traci.close()

    def set_stats(self):
        self.stats["total_waiting_time"] = sum(self.waiting_times.values())
        self.stats["halting_vehicle_count"] = self.count_halting_vehicles()[1]
        self.stats["co2_emission_total"] = round(self.co2_emission/1000)
        self.stats["avg_queue_length"] = self.sum_queue/self.max_step
        self.stats["avg_wait_time"] = self.stats["total_waiting_time"]/self.count_halting_vehicles()[0],

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
    simulation = Simulation('sumo-gui', 'sumo_files/osm.sumocfg', 200)
    simulation.run()
    print(simulation.stats)