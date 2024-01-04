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
       #old_action = -1

        # routes = self._environment.generate_routes()

        for currentStep in range(self.max_step):
            
            # # Model + Environment
            # state = None
            # # state = self._environment.get_state()
            # action = self._model.choose_action(state)

            # if action != old_action:
            #     steps_to_do_yellow_phase_1 = self._environment.set_yellow_phase(old_action)
            #     self._run_steps(steps_to_do_yellow_phase_1, currentStep)

            #     steps_to_do_red_phase = self._environment.set_red_phase()
            #     self._run_steps(steps_to_do_red_phase, currentStep)

            #     steps_to_do_yellow_phase_2 = self._environment.set_yellow_phase(action)
            #     self._run_steps(steps_to_do_yellow_phase_2, currentStep)

            #     steps_to_do_green_phase = self._environment.set_green_phase(action)
            #     self._run_steps(steps_to_do_green_phase, currentStep)
            # else:
            self._run_steps(1, currentStep)
            

            # STATS
            vehicles = traci.vehicle.getIDList()
            for vehicle in vehicles:
                self.get_waiting_time(vehicle)
                self.co2_emission += self.get_co2_emission(vehicle)
                self.sum_queue += sum(self.get_queue_length())

            # UPDATE
            
            #old_action = action

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
        self.stats["avg_wait_time"] = self.stats["total_waiting_time"]/self.count_halting_vehicles()[0]
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


class CrossIntersection():
    # funktion: _get_state
    # funktion: skift lys (tager id på lyskryds som input)
    # funktion: skift til action nr. (tager action nr. som input) - bruger skift lys funktion

    def __init__(self, sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps):
        self.sumo_path = 'sumo_files/osm.sumocfg'
        self.sumo_mode = sumo_mode

        self.min_green_phase_steps = min_green_phase_steps
        self.yellow_phase_steps = yellow_phase_steps
        self.red_phase_steps = red_phase_steps

        self.num_actions = 2

        self.actions_def = {
            0: {'green_phase_index': 0, 'yellow_phase_index': 1},
            1: {'green_phase_index': 3, 'yellow_phase_index': 4}
        }

        self.phases = ["rrrrGGggrrrrGGgg", # NS green phase
                "rrrryyyyrrrryyyy", # NS yellow phase
                "rrrrrrrrrrrrrrrr", # all red phase
                "GGggrrrrGGggrrrr",# EW green phase
                "yyyyrrrryyyyrrrr"] # EW yellow phase

        self.lane = ["-125514713_0", "-125514711_0", "-548975769_0", "125514709_0"]
        self.traffic_light_system_id = "24960712"
        

    def run_env(self):
        traci.start([self.sumo_mode, "-c", self.sumo_path])
        self._set_phases()

    def _get_phases(self):
        tls_definitions = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.traffic_light_system_id)
        for logic in tls_definitions:
            print(f"Traffic Light {self.traffic_light_system_id} has the following phases in program {logic.programID}:")
            for phase in logic.getPhases():
                print(f"Phase index {logic.getPhases().index(phase)}: {phase.state} with duration {phase.duration} seconds")

    def _set_phases(self):

        logic = traci.trafficlight.Logic(
            programID="newProgramID",
            type=0, 
            currentPhaseIndex=0,
            phases=self.phases
        )

        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.traffic_light_system_id, logic)
  
    def set_yellow_phase(self, action):
        phase_index = self.actions_def[action]['yellow_phase_index']
        traci.trafficlight.setPhase(self.traffic_light_system_id, phase_index)

        return self.yellow_phase_steps

    def set_red_phase(self):
        phase_index = 2
        traci.trafficlight.setPhase(self.traffic_light_system_id, phase_index)
        return self.red_phase_steps

    def set_green_phase(self, action):
        phase_index = self.actions_def[action]['green_phase_index']
        traci.trafficlight.setPhase(self.traffic_light_system_id, phase_index)

        return self.min_green_phase_steps

    # def set_green_phase(self, action_number):
    #     """
    #     Activate the correct green light combination in sumo
    #     """
    #     if action_number == 0:
    #         traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
    #     elif action_number == 1:
    #         traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
    #     elif action_number == 2:
    #         traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
    #     elif action_number == 3:
    #         traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)



    # def _get_state(self):
    #     # Determine which lanes are occupied
    #     occupied_lanes = []
    #     for lane in self.lane:
    #         if traci.edge.getLastStepHaltingNumber(lane) > 0:
    #             occupied_lanes.append(lane)

    #     # Represent the state as a binary number (0 or 1)
    #     if occupied_lanes == ["lane:-125514711_0", "lane:125514709_0"]:
    #         state = 0  # North-south advance state
    #     elif occupied_lanes == ["lane:-125514713_0", "lane:-548975769_0"]:
    #         state = 1  # West-east advance state
    #     else:
    #         state = None  # Invalid state

    #     return state

    def get_traffic_light_state(self):

        state_string = traci.trafficlight.getRedYellowGreenState(self.traffic_light_system_id)
        light_states = {
            'red': [],
            'yellow': [],
            'green': []
        }
        
        # Loop through the state string and populate the dictionary
        for index, state in enumerate(state_string):
            if state == 'r':
                light_states['red'].append(index)
            elif state == 'y':
                light_states['yellow'].append(index)
            elif state.lower() == 'g':  # 'G' or 'g' can be used for green
                light_states['green'].append(index)
        
        return light_states

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