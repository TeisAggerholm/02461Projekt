import traci
import os
import numpy as np
import random

class CrossIntersection():
    # funktion: _get_state
    # funktion: skift lys (tager id på lyskryds som input)
    # funktion: skift til action nr. (tager action nr. som input) - bruger skift lys funktion

    def __init__(self, sumo_mode, min_green_phase_steps, yellow_phase_steps, red_phase_steps, max_step):
        self.sumo_path = 'sumo_files/osm.sumocfg'
        self.sumo_mode = sumo_mode
        self.max_step = max_step

        self.min_green_phase_steps = min_green_phase_steps
        self.yellow_phase_steps = yellow_phase_steps
        self.red_phase_steps = red_phase_steps

        self.num_actions = 2

        self.actions_def = {
            0: {'green_phase_index': 0, 'yellow_phase_index': 1},
            1: {'green_phase_index': 3, 'yellow_phase_index': 4}
        }

        self.phases = [
            traci.trafficlight.Phase(10, "rrrrGGggrrrrGGgg"), # NS green phase
            traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy"), # NS yellow phase
            traci.trafficlight.Phase(2, "rrrrrrrrrrrrrrrr"), # all red phase
            traci.trafficlight.Phase(10, "GGggrrrrGGggrrrr"), # EW green phase
            traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"), # EW yellow phase
        ]

        self.lane = ["-125514713_0", "-125514711_0", "-548975769_0", "125514709_0"]
        self.traffic_light_system_id = "24960712"
        

    def run_env(self):
        self.route_generate(self.max_step, 0.75)
        traci.start([self.sumo_mode, "-c", self.sumo_path])
        self._set_phases()

    def route_generate(self, max_step, percentage_straight): 
        #Anvend ved kontrol: seed. 
        
        #Antal biler: 
        n_cars = np.random.normal(max_step/60*66.667,30)

        #Hvilken fordeling anvendes til bilernes tider? 
        tider = np.random.weibull(2, int(n_cars))
        tider.sort()

        #Normalisering ift. max_steps 
        new_times = []
        for tid in tider: 
            new_times.append(tid / tider[-1] * max_step)



        with open("sumo_files/osm.passenger.trips.xml", "w") as routes:
            print("""<?xml version="1.0" encoding="UTF-8"?>

            <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                  
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />) 
            
            <route id="N2W" edges="-125514711 125514713"/>
            <route id="N2S" edges="-125514711 -125514709"/>
            <route id="N2E" edges="-125514711 -548975769"/>     
            <route id="W2N" edges="-125514713 125514711"/>
            <route id="W2E" edges="-125514713 -548975769"/>
            <route id="W2S" edges="-125514711 -125514709"/>
            <route id="S2W" edges="125514709 125514713"/>
            <route id="S2N" edges="125514709 125514711"/>
            <route id="S2E" edges="-125514711 -548975769"/>
            <route id="E2S" edges="548975769 -125514709"/>
            <route id="E2W" edges="548975769 125514713"/>
            <route id="E2N" edges="548975769 125514711"/>
            """, file=routes)

            for car, departure in enumerate(new_times): 
                if np.random.rand(1)<percentage_straight: 
                    random_int = np.random.randint(1,5)
                    if random_int == 1: 
                        print(f'<vehicle id="N2S{car}" type="standard_car" route="N2S" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 2: 
                        print(f'<vehicle id="W2E{car}" type="standard_car" route="W2E" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 3: 
                        print(f'<vehicle id="S2N{car}" type="standard_car" route="S2N" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)  
                    if random_int == 4: 
                        print(f'<vehicle id="E2W{car}" type="standard_car" route="E2W" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)

                else: 
                    random_int = np.random.randint(1,9)
                    if random_int == 1: 
                        print(f'<vehicle id="N2W{car}" type="standard_car" route="N2W" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 2: 
                        print(f'<vehicle id="N2E{car}" type="standard_car" route="N2E" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 3: 
                        print(f'<vehicle id="W2N{car}" type="standard_car" route="W2N" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)  
                    if random_int == 4: 
                        print(f'<vehicle id="W2S{car}" type="standard_car" route="W2S" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 5: 
                        print(f'<vehicle id="S2W{car}" type="standard_car" route="S2W" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 6: 
                        print(f'<vehicle id="S2E{car}" type="standard_car" route="S2E" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)
                    if random_int == 7: 
                        print(f'<vehicle id="E2N{car}" type="standard_car" route="E2N" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)  
                    if random_int == 8: 
                        print(f'<vehicle id="E2S{car}" type="standard_car" route="E2S" depart="{departure}" departLane="random" departSpeed="10" />', file=routes)

            print("</routes>", file=routes)
        

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
