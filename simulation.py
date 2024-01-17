import traci
import os
from models.dqn import DQN
import numpy as np
import torch

class Simulation:
    def __init__(self, max_step, _environment, _model, final_score_weights, episodes, memory, epochs):
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
        self.batch_size = 100
        self._waiting_times = {}
        self.overall_reward_queue_length = 0
        self.epochs = epochs

    def run(self):
        self._environment.run_env()
        old_action = -1
        last_executed_action = -1
        last_wait = 0
        
        previous = {"state_list": 0, "action": 0, "reward": 0}

        while self._currentStep < self.max_step:

            # STATE
            state = self.get_state()   
           
            #ADD TO MEMORY
            if old_action != -1:
                exp = (previous["state_list"], previous["action"], previous["reward"], state.tolist(), 0)
                #print("State_list", exp) #PRINT FOR STATE_LIST pr STEP
                self.memory.add_experience(exp)


            # ACTION
            action = self._model.choose_action(state, self._currentStep)
            self._environment.set_specific_phase(action)

            # REWARD
            current_total_wait = self.get_waiting_time()
            reward = last_wait - current_total_wait
            
            # UPDATE        
            traci.simulationStep()    
            self._currentStep += 1     

            last_wait = current_total_wait

            # ONLY NEGATIVE REWARD FOR TOTAL REWARD
            if reward < 0: 
                self.overall_reward += reward

            # OVERALL_REWARD_QUEUE_LENGTH   
            self.overall_reward_queue_length += sum(state.tolist())
            
            # Save previous state
            previous["state_list"] = state.tolist()
            previous["action"] = action
            previous["reward"] = reward
            old_action = action
            
        traci.close()


        self.overall_reward
        
        self._model.epsilon_dec_fun()
        for i in range(self.epochs): 
            batch = self.memory.get_batch(50)
            self._model.train(batch)
        print("---DONE TRAINING---")

    
    def queue_length_state(self): 
        Halt_N = traci.edge.getLastStepHaltingNumber("-125514711")
        Halt_S = traci.edge.getLastStepHaltingNumber("125514709")
        Halt_E = traci.edge.getLastStepHaltingNumber("548975769")
        Halt_W = traci.edge.getLastStepHaltingNumber("-125514713")
          
        return torch.Tensor([Halt_N, Halt_S, Halt_E, Halt_W])


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
    
    def get_state(self): 

        car_list = traci.vehicle.getIDList()    
        state = np.zeros(80)
        


        for car in car_list: 
            
            route_id = traci.vehicle.getRouteID(car)
            lane_pos = traci.vehicle.getLanePosition(car)
            lane_id = traci.vehicle.getLaneID(car)
            lane_pos = 100 - lane_pos
            
            if lane_pos < 10:
                lane_cell = 0
            elif lane_pos < 20:
                lane_cell = 1
            elif lane_pos < 30:
                lane_cell = 2
            elif lane_pos < 40:
                lane_cell = 3
            elif lane_pos < 50:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 70:
                lane_cell = 6
            elif lane_pos < 80:
                lane_cell = 7
            elif lane_pos < 90:
                lane_cell = 8
            elif lane_pos < 100:
                lane_cell = 9
            
            #Finding lan of the car: 
        
            if lane_id == "-125514711_0": #NORTH 
                lane_group = 0

                if route_id == "N2E":
                    lane_group = 1

            elif lane_id == "125514709_0": #South 
                lane_group = 2
    
                if route_id == "S2W":
                    lane_group = 3
                    
            elif lane_id == "548975769_0": #EAST 
                lane_group = 4

                if route_id == "E2S":
                    lane_group = 5

            elif lane_id == "-125514713_0": #WEST 
                lane_group = 6

                if route_id == "W2N": 
                    lane_group = 7
            else:
                lane_group = -1


            if lane_group >= 1 and lane_group <= 3:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"
                
        return torch.Tensor(state.tolist())
        
