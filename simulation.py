import traci
import os
from models.dqn import Memory, Experience, DQN
import numpy as np
class Simulation:
    def __init__(self, max_step, _environment, _model, final_score_weights, episodes, memory):
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
        self.memory = memory
        self.overall_reward = 0
        self.batch_size = 64
        self._waiting_times = {}

    def run(self):
        self._environment.run_env()
        old_action = -1
        last_executed_action = -1
        last_wait = 0
        
        previous = {"state_list": 0, "action": 0, "reward": 0}

        while self._currentStep < self.max_step:
            state_list = self.get_queue_length()
            
           # state_github = self.getGithubState()
            
            #ADD TO MEMORY
            if old_action != -1:
                exp = (previous["state_list"], previous["action"], previous["reward"], state_list, 0)
                #print("State_list", exp) #PRINT FOR STATE_LIST pr STEP
                self.memory.add_experience(exp)


            # Action
            action = self._model.choose_action(state_list, self._currentStep)
            isActionable = self._environment.isActionable()
            
            if isActionable:
                if last_executed_action == -1:
                    self._environment.push_green_phase(action)
                    last_executed_action = action

                elif action != last_executed_action:
                    self._environment.push_yellow_phase(last_executed_action)
                    self._environment.push_red_phase()
                    self._environment.push_yellow_phase(action)
                    self._environment.push_green_phase(action)
                    last_executed_action = action

            #REWARD
            current_total_wait = self.get_waiting_time()
            reward = last_wait - current_total_wait
            # UPDATE        
            self._environment.set_lights()
            traci.simulationStep()    
            self._currentStep += 1     
            self._environment.increment_steps_in_current_phase()
            self._environment.update_current_phase()
            last_wait = current_total_wait

            if reward < 0: 
                self.overall_reward += reward

            previous["state_list"] = state_list
            previous["action"] = action
            previous["reward"] = reward
            old_action = action
            
            # Train
           # if len(self.memory._experiences) > self.batch_size and self._currentStep % 10 == 0:
               # batch = self.memory.get_batch(self.batch_size)
                #print("-----CURRENT LOSS----",self._model.train(batch))

        traci.close()
        self._model.epsilon_dec_fun()
        for i in range(400): 
            batch = self.memory.get_batch(50)
            self._model.train(batch)
        print("---DONE TRAINING---")
    
    def getGithubState(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_len = traci.lane.getLength(lane_id)
            lane_pos = lane_len - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located 
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state

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

    def get_waiting_time(self):
        incoming_roads = ["-125514711","125514709","548975769","-125514713"]
        cars = traci.vehicle.getIDList()
        for car_id in cars: 
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id) #Get the road of the car
            if road_id in incoming_roads: #Only waiting times of incoming road.
                self._waiting_times[car_id] = wait_time
            else: 
                if car_id in self._waiting_times: # A car that has cleared the intersection. 
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time
     
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
