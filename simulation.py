import traci
import os
from models.dqn import DQN
import numpy as np
import torch

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

            # STATE
            state = self.queue_length_state()
                        
            #ADD TO MEMORY
            if old_action != -1:
                exp = (previous["state_list"], previous["action"], previous["reward"], state.tolist(), 0)
                #print("State_list", exp) #PRINT FOR STATE_LIST pr STEP
                self.memory.add_experience(exp)


            # ACTION
            action = self._model.choose_action(state, self._currentStep)
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

            # REWARD
            current_total_wait = self.get_waiting_time()
            reward = last_wait - current_total_wait
            # UPDATE        
            self._environment.set_lights()
            traci.simulationStep()    
            self._currentStep += 1     
            self._environment.increment_steps_in_current_phase()
            self._environment.update_current_phase()
            last_wait = current_total_wait

            # ONLY NEGATIVE REWARD FOR TOTAL REWARD
            if reward < 0: 
                self.overall_reward += reward

            previous["state_list"] = state.tolist()
            previous["action"] = action
            previous["reward"] = reward
            old_action = action
            
        traci.close()

        
        self._model.epsilon_dec_fun()
        for i in range(400): 
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