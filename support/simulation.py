import traci
import os

class Simulation:
    def __init__(self, max_step, _environment, _model):
        self.max_step = max_step
        self._environment = _environment
        self._model = _model
        self.sum_queue = 0
        self.co2_emission = 0
        self.waiting_times = {}
        self.stats = {}

    def run(self):
        self._environment.run_env()
        old_action = -1

        # routes = self._environment.generate_routes()

        for currentStep in range(self.max_step):
            
            # Model + Environment
            state = None
            # state = self._environment.get_state()
            action = self._model.choose_action(state)

            if action != old_action:
                steps_to_do_yellow_phase_1 = self._environment.set_yellow_phase(old_action)
                self._run_steps(steps_to_do_yellow_phase_1, currentStep)

                steps_to_do_red_phase = self._environment.set_red_phase()
                self._run_steps(steps_to_do_red_phase, currentStep)

                steps_to_do_yellow_phase_2 = self._environment.set_yellow_phase(action)
                self._run_steps(steps_to_do_yellow_phase_2, currentStep)

                steps_to_do_green_phase = self._environment.set_green_phase(action)
                self._run_steps(steps_to_do_green_phase, currentStep)
            else:
                  self._run_steps(1, currentStep)
            

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

    def _run_steps(self, steps_to_do, currentStep):

        if(currentStep + steps_to_do) >= self.max_step:
            steps_to_do = self.max_step - currentStep # do not do more steps than the maximum allowed number of steps

        while steps_to_do > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            currentStep += 1 # update the step counter
            steps_to_do -= 1

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