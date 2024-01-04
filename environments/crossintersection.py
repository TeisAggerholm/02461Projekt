import traci
import os


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
    sumo_config_path = 'sumo_files/osm.sumocfg'

    traci.start(["sumo-gui", "-c", sumo_config_path])

    environment = CrossIntersection(3)
    environment._set_phases()
    environment._get_phases()

    for step in range(200):
        traci.simulationStep()
        print(step)
        environment.get_traffic_light_state()


    traci.close()

















#     def _choose_action(self, state, epsilon):
#         # Use the model to predict the best action given the state
#         action = self._Model.predict_one(state)

#         # Modify the action selection based on the number of occupied lanes
#         if state == 0 and len(occupied_lanes) == 1:
#             # If only one lane is occupied, prioritize the opposite lane
#             action = (action + 1) % 2

#         return action


#     def _set_yellow_phase(self, old_action):
#         # Determine the yellow phase code based on the previous action
#         yellow_phase_code = old_action * 2 + 1

#         # Activate the yellow light combination in sumo
#         traci.trafficlight.setPhase("TL", yellow_phase_code)


# class environment_4_arms:
#     # Skal give hvilket antal mulige actions der er.
#     # Skal håndtere hvad der sker ved hver action og hermed ændre lyskrydsene.



#     def __init__(self):
#         pass