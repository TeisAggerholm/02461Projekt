import traci
import os
from models.dqn import Memory, Experience

class Simulation:
    def __init__(self, max_step, _environment, _model, final_score_weights, episodes):
        self.episodes = episodes
        self.max_step = max_step
        self._environment = _environment
        self._model = _model
        self._currentStep = 0
        self.sum_queue = 0
        self.co2_emission = 0
        self.waiting_times = {}
        self.stats = {}
        self.final_score_weights = final_score_weights
        self.memory = Memory(1000)
        self.overall_reward = 0

    def run(self):
        self._environment.run_env()
        old_action = -1

        previous = {"state_list": 0, "action": 0, "reward": 0}

        while self._currentStep < self.max_step:
            state_list = self.get_queue_length()
            
            #ADD TO MEMORY
            if old_action == 0 or old_action == 1:
                exp = (previous["state_list"], previous["action"], previous["reward"], state_list)
                self.memory.add_experience(exp)


            action = self._model.choose_action(state_list)

            # if old_action == -1:
            #     steps_to_do_green_phase = self._environment.set_green_phase(action)
            #     self._run_steps(steps_to_do_green_phase)
            # elif action != old_action:
            #     steps_to_do_yellow_phase_1 = self._environment.set_yellow_phase(old_action)
            #     self._run_steps(steps_to_do_yellow_phase_1)

            #     steps_to_do_red_phase = self._environment.set_red_phase()
            #     self._run_steps(steps_to_do_red_phase)

            #     steps_to_do_yellow_phase_2 = self._environment.set_yellow_phase(action)
            #     self._run_steps(steps_to_do_yellow_phase_2)

            #     steps_to_do_green_phase = self._environment.set_green_phase(action)
            #     self._run_steps(steps_to_do_green_phase)

            self._run_steps(1)
            
            
            # UPDATE
            reward = - sum(self.get_queue_length())
            self.overall_reward -= sum(self.get_queue_length())

            previous["state_list"] = state_list
            previous["action"] = action
            previous["reward"] = reward
            
            old_action = action
            
            if len(self.memory._experiences) > 10:
                batch = self.memory.get_batch(10)
                self._model.train(batch)

        self.set_stats()
        traci.close()
         
    def _run_steps(self, steps_to_do):

        if(self._currentStep + steps_to_do) >= self.max_step:
            steps_to_do = self.max_step - self._currentStep # do not do more steps than the maximum allowed number of steps

        while steps_to_do > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._get_vehicle_stats()
            self._currentStep += 1 # update the step counter
            steps_to_do -= 1

    def _get_vehicle_stats(self):
        # STATS
        vehicles = traci.vehicle.getIDList()
        vehicle_waiting_times = []

        for vehicle in vehicles:
            vehicle_waiting_times.append(self.get_waiting_time(vehicle))
            self.co2_emission += self.get_co2_emission(vehicle)
            self.sum_queue = sum(self.get_queue_length())
        
        return {"wait_times": vehicle_waiting_times, "total_co2": self.co2_emission, "queues": self.get_queue_length()}

    def set_stats(self):
        self.stats["total_waiting_time"] = sum(self.waiting_times.values())
        self.stats["halting_vehicle_count"] = self.count_halting_vehicles()[1]
        self.stats["co2_emission_total"] = round(self.co2_emission/1000)
        self.stats["avg_queue_length_per_step"] = self.sum_queue/self.max_step
        self.stats["avg_wait_time"] = round(self.stats["total_waiting_time"]/self.count_halting_vehicles()[0],2)
        self.stats["vehicle_count"] = self.count_halting_vehicles()[0]

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

        return wait_time
     
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
    
    def calc_overall_score(self):
        overall_score = self.final_score_weights["total_waiting_time"] * self.stats["total_waiting_time"]
        return overall_score
    




class Stats:
    def __init__(self):
        pass

    def isHalted(self, vehicle_id):
        return traci.vehicle.getSpeed(vehicle_id) < 0.1 # speed under 0.1
     
if __name__ == '__main__':
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    environments_path = os.path.join(project_root, 'environments')
    sys.path.append(environments_path)
    from crossintersection import CrossIntersection # Det virker selvom der er linje under  crossintersection - det er en vscode bug

    simulation = Simulation(200,CrossIntersection("sumo",10,2,2),{})
    print("These are the total stats of the run")
    simulation.run()
    print(simulation.stats)
    print(simulation.calc_overall_score())
