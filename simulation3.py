import traci
import os
import torch
import numpy as np

class Simulation:
    def __init__(self, max_step, _environment, _model, _memory, epochs, batch_size):
        self.max_step = max_step
        self._environment = _environment
        self._model = _model
        self._memory = _memory
        self.epochs = epochs

        self._currentStep = 0
        self.overall_reward = 0
        
        self._waiting_times = {}

        self.batch_size = batch_size

    def run(self):
        self._environment.run_env()

        old_action = -1
        old_total_wait  = 0
        old_state = -1
        
        
        while self._currentStep < self.max_step:
            state = self.get_state()

            current_total_wait = self.get_waiting_time()
            reward = old_total_wait - current_total_wait

            action = self._model.choose_action(state, self._currentStep)

            if self._currentStep != 0:
                self._memory.add_experience((old_state.tolist(), old_action, reward, state.tolist(),0))
                print("---ADDED TO MEMORY----", f'STEP: {self._currentStep}')
                
            if old_action == -1:
                steps_to_do_green_phase = self._environment.set_green_phase(action)
                self._run_steps(steps_to_do_green_phase)
            elif action != old_action:
                steps_to_do_yellow_phase_1 = self._environment.set_yellow_phase(old_action)
                self._run_steps(steps_to_do_yellow_phase_1)

                steps_to_do_red_phase = self._environment.set_red_phase()
                self._run_steps(steps_to_do_red_phase)

                steps_to_do_yellow_phase_2 = self._environment.set_yellow_phase(action)
                self._run_steps(steps_to_do_yellow_phase_2)

                steps_to_do_green_phase = self._environment.set_green_phase(action)
                self._run_steps(steps_to_do_green_phase)
            else:
                steps_to_do_green_phase = self._environment.set_green_phase(action)
                self._run_steps(steps_to_do_green_phase)

            
            # STATS

            # UPDATE
            old_action = action
            old_state = state
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly          
            if reward < 0: 
                self.overall_reward += reward
        
        traci.close()

        for _ in range(self.epochs):
            batch = self._memory.get_batch(self.batch_size)
            self._model.train(batch)
            print("----DONE TRAINING----")

    def _run_steps(self, steps_to_do):

        if(self._currentStep + steps_to_do) >= self.max_step:
            steps_to_do = self.max_step - self._currentStep # do not do more steps than the maximum allowed number of steps

        while steps_to_do > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._currentStep += 1 # update the step counter
            steps_to_do -= 1
    
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

